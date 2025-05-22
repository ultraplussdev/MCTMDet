import torch
import torch.nn as nn
from timm.models.registry import register_model
import math
from timm.models.layers import trunc_normal_, DropPath, LayerNorm2d
from timm.models._builder import resolve_pretrained_cfg

try:
    from timm.models._builder import _update_default_kwargs as update_args
except:
    from timm.models._builder import _update_default_model_kwargs as update_args
from timm.models.vision_transformer import Mlp, PatchEmbed
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat
from pathlib import Path

class MambaTransfomer(nn.Module):
    """
    MambaVision layer"
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 transformer_blocks=[],

                 ):
        """
        Args:
            dim: feature size dimension.
            depth: number of layers in each stage.
            window_size: window size in each stage.
            conv: bool argument for conv stage flag.
            downsample: bool argument for down-sampling.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: drop path rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
            layer_scale_conv: conv layer scaling coefficient.
            transformer_blocks: list of transformer blocks.
        """

        super().__init__()

        self.blocks = nn.ModuleList([MambaAttentionBlock(dim=dim,
                                           counter=i,
                                           transformer_blocks=transformer_blocks,
                                           num_heads=num_heads,
                                           mlp_ratio=mlp_ratio,
                                           qkv_bias=qkv_bias,
                                           qk_scale=qk_scale,
                                           drop=drop,
                                           attn_drop=attn_drop,
                                           drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                           layer_scale=layer_scale)
                                     for i in range(depth)])
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, LayerNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    def forward(self, x):
        B, L, H, W, C = x.shape
        pos_3d = get_3d_sin_cos_pos_embed(C, L, H, W).to(x.device)  # [L,H,W,C]
        pos_3d = pos_3d.unsqueeze(0).expand(B, -1, -1, -1, -1)  # => [B,L,H,W,C]
        x = x + pos_3d
        for _, blk in enumerate(self.blocks):
            x = blk(x)
        return x
class MambaAttentionBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 counter,
                 transformer_blocks,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 Mlp_block=Mlp,
                 layer_scale=None,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if counter in transformer_blocks:
            self.mixer = Attention_BLHWC(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                norm_layer=norm_layer,
            )
        else:
            self.mixer = MambaVisionBlock(d_model=dim,
                                          d_state=8,
                                          d_conv=3,
                                          expand=1
                                          )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class MambaVisionBlock(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=1,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,
            layer_idx=None,
            device=None,
            dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.d_space = self.d_inner // 4
        self.d_time = self.d_inner // 4
        self.d_conv1d = self.d_inner // 2
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.space_x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.time_x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.space_dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        self.time_dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.space_dt_proj.weight, dt_init_std)
            nn.init.constant_(self.time_dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.space_dt_proj.weight, -dt_init_std, dt_init_std)
            nn.init.uniform_(self.time_dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.space_dt_proj.bias.copy_(inv_dt)
            self.time_dt_proj.bias.copy_(inv_dt)
        self.space_dt_proj.bias._no_reinit = True
        self.time_dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.space_A_log = nn.Parameter(A_log)
        self.space_A_log._no_weight_decay = True
        self.space_D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.space_D._no_weight_decay = True
        self.time_A_log = nn.Parameter(A_log)
        self.time_A_log._no_weight_decay = True
        self.time_D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.time_D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.space_out = nn.Linear(self.d_inner, self.d_space,bias=bias, **factory_kwargs)
        self.time_out = nn.Linear(self.d_inner, self.d_time,bias=bias, **factory_kwargs)
        self.space_conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            **factory_kwargs,
        )
        self.time_conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            **factory_kwargs,
        )
        self.branch_conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_conv1d,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_conv1d,
            **factory_kwargs,
        )

    def forward(self, x):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        B, L, H,W, C = x.shape
        x = self.in_proj(x)
        seqlen = L*H*W
        # xz = rearrange(xz, "b l d -> b d l")
        # x, z = xz.chunk(2, dim=1)
        space_x = rearrange(x, 'b l h w c -> b c (l h w)')
        space_A = -torch.exp(self.space_A_log.float())
        space_x = F.silu(F.conv1d(input=space_x, weight=self.space_conv1d.weight, bias=self.space_conv1d.bias, padding='same',
                            groups=self.d_inner))
        space_x_dbl = self.space_x_proj(rearrange(space_x, "b d l -> (b l) d"))
        dt, B, C = torch.split(space_x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.space_dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        space_y = selective_scan_fn(space_x,
                              dt,
                              space_A,
                              B,
                              C,
                              self.space_D.float(),
                              z=None,
                              delta_bias=self.space_dt_proj.bias.float(),
                              delta_softplus=True,
                              return_last_state=None)
        space_y = rearrange(space_y, "b d (l h w) -> b l (h w) d",l=L,h=H,w=W)

        space_y = self.space_out(space_y)
        time_x = rearrange(x, 'b l h w c -> b c (h w l)')
        time_A = -torch.exp(self.time_A_log.float())
        time_x = F.silu(
            F.conv1d(input=time_x, weight=self.time_conv1d.weight, bias=self.time_conv1d.bias, padding='same',
                     groups=self.d_inner))
        time_x_dbl = self.time_x_proj(rearrange(time_x, "b d l -> (b l) d"))
        dt, B, C = torch.split(time_x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.time_dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        time_y = selective_scan_fn(time_x,
                              dt,
                              time_A,
                              B,
                              C,
                              self.time_D.float(),
                              z=None,
                              delta_bias=self.time_dt_proj.bias.float(),
                              delta_softplus=True,
                              return_last_state=None)
        time_y = rearrange(time_y, "b d (h w l) -> b l (h w) d",l= L,h=H,w=W)
        time_y = self.time_out(time_y)
        conv_x = rearrange(x, 'b l h w c -> b c (l h w)')
        conv_y =F.silu(F.conv1d(input=conv_x, weight=self.branch_conv1d.weight, bias=self.branch_conv1d.bias, padding='same',
                            groups=self.d_conv1d))
        conv_y = rearrange(conv_y,"b d (l h w) -> b l (h w) d",l= L,h=H,w=W)
        y = torch.cat([space_y, time_y,conv_y], dim=-1)
        out = self.out_proj(y)
        out = rearrange(out,"b l (h w) d -> b l h w d",l= L,h=H,w=W)
        return out

class Attention_BLHWC(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=True,
            qk_norm=True,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B,L,H,W,C = x.shape
        x= rearrange(x,'b l h w c-> b (l h w) c',l=L,h=H,w=W)
        N = L*H*W
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x= rearrange(x,'b (l h w) c-> b l h w c',l=L,h=H,w=W)
        return x

def build_1d_sin_cos(n, dim):

    pe = torch.zeros(n, dim)
    position = torch.arange(n, dtype=torch.float32).unsqueeze(1)  # [n, 1]
    div_term = torch.exp(torch.arange(0, (dim + 1) // 2, dtype=torch.float32) * -(math.log(10000.0) / dim))

    pe[:, 0::2] = torch.sin(position * div_term[:pe[:, 0::2].shape[1]])
    pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])

    return pe


def get_3d_sin_cos_pos_embed(embed_dim, L, H, W):

    c_l = embed_dim // 3
    c_h = embed_dim // 3
    c_w = embed_dim - c_l - c_h


    pos_l = build_1d_sin_cos(L, c_l)  # [L, c_l]
    pos_h = build_1d_sin_cos(H, c_h)  # [H, c_h]
    pos_w = build_1d_sin_cos(W, c_w)  # [W, c_w]

    pos_l = pos_l[:, None, None, :].expand(L, H, W, c_l)
    pos_h = pos_h[None, :, None, :].expand(L, H, W, c_h)
    pos_w = pos_w[None, None, :, :].expand(L, H, W, c_w)

    # cat => [L,H,W, c_l + c_h + c_w] = [L,H,W, embed_dim]
    pos_3d = torch.cat([pos_l, pos_h, pos_w], dim=3)

    return pos_3d