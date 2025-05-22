import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath


# -------------------------
# BaseConv
# -------------------------
class BaseConv(nn.Module):
    """A Conv2d -> BatchNorm -> Activation block."""
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if act == "silu" else nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# -------------------------
# CrossAttention
# -------------------------
class CrossAttention(nn.Module):
    def __init__(self, q_dim=512, kv_dim=512, out_dim=512, num_heads=8,
                 qkv_bias=False, qk_norm=True, attn_drop=0., proj_drop=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        assert out_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(q_dim, out_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(kv_dim, out_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(kv_dim, out_dim, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(out_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q_input, kv_input):
        B, N_q, _ = q_input.shape
        B, N_kv, _ = kv_input.shape

        q = self.q_proj(q_input).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(kv_input).reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(kv_input).reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q = self.q_norm(q)
        k = self.k_norm(k)

        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        x = attn.transpose(1, 2).reshape(B, N_q, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0,
                 qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, act_layer=nn.GELU):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            q_dim=dim, kv_dim=dim, out_dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            act_layer(),
            nn.Dropout(proj_drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(proj_drop)
        )

    def forward(self, x, kv):
        x = x + self.drop_path(self.attn(self.norm1(x), kv))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x





# -------------------------
# EncoderMambaV2
# -------------------------
class DecoderMamba(nn.Module):
    def __init__(self, in_channels=[256, 512, 1024], context_dim=None):
        super().__init__()
        if context_dim is None:
            context_dim = in_channels[1]
        self.low_channels = in_channels[0]
        self.mid_channels = in_channels[1]
        self.high_channels = in_channels[2]
        self.context_dim = context_dim

        self.context_to_low = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            BaseConv(self.context_dim, self.low_channels, 3, 1)
        )
        self.context_to_mid = BaseConv(self.context_dim, self.mid_channels, 1, 1)
        self.context_to_high = nn.Sequential(
            BaseConv(self.context_dim, self.context_dim, 3, 1),
            BaseConv(self.context_dim, self.high_channels, 3, 2)
        )

        self.low_block = CrossAttentionBlock(dim=self.low_channels)
        self.mid_block = CrossAttentionBlock(dim=self.mid_channels)
        self.high_block = CrossAttentionBlock(dim=self.high_channels)

    def forward(self, features, context):
        low, mid, high = features
        B, L, H_mid, W_mid, C = context.shape
        H_low, W_low = low.shape[3], low.shape[4]
        H_high, W_high = high.shape[3], high.shape[4]

        context_2d = rearrange(context, 'b l h w c -> (b l) c h w')
        context_low = self.context_to_low(context_2d)
        context_mid = self.context_to_mid(context_2d)
        context_high = self.context_to_high(context_2d)

        low_flat = rearrange(low, 'b l c h w -> (b l) (h w) c')
        mid_flat = rearrange(mid, 'b l c h w -> (b l) (h w) c')
        high_flat = rearrange(high, 'b l c h w -> (b l) (h w) c')

        ctx_low_flat = rearrange(context_low, 'bl c h w -> bl (h w) c')
        ctx_mid_flat = rearrange(context_mid, 'bl c h w -> bl (h w) c')
        ctx_high_flat = rearrange(context_high, 'bl c h w -> bl (h w) c')

        low_enhanced = self.low_block(low_flat, ctx_low_flat) + low_flat
        mid_enhanced = self.mid_block(mid_flat, ctx_mid_flat) + mid_flat
        high_enhanced = self.high_block(high_flat, ctx_high_flat) + high_flat

        low_enhanced = rearrange(low_enhanced, 'bl (h w) c -> bl c h w', h=H_low, w=W_low)
        mid_enhanced = rearrange(mid_enhanced, 'bl (h w) c -> bl c h w', h=H_mid, w=W_mid)
        high_enhanced = rearrange(high_enhanced, 'bl (h w) c -> bl c h w', h=H_high, w=W_high)

        return [low_enhanced, mid_enhanced, high_enhanced]
