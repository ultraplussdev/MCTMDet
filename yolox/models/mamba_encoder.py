import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F


# -------------------------
# BaseConv
# -------------------------
class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride,
                              padding=pad, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if act == "silu" else nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# -------------------------
# Positional Encoding
# -------------------------
def get_2d_sincos_pos_embed(embed_dim, h, w):
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w = torch.arange(w, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing='xy')  # [2, H, W]
    grid = torch.stack(grid, dim=0)  # [2, H, W]

    pos_embed = []
    for i in range(2):  # for h and w
        pos = grid[i].reshape(-1)
        dim_half = embed_dim // 2
        omega = torch.arange(dim_half, dtype=torch.float32) / dim_half
        omega = 1. / (10000 ** omega)  # [D/2]
        out = pos[:, None] * omega[None, :]  # [H*W, D/2]
        emb_sin = torch.sin(out)
        emb_cos = torch.cos(out)
        emb = torch.cat([emb_sin, emb_cos], dim=1)  # [H*W, D]
        pos_embed.append(emb)
    return torch.cat(pos_embed, dim=1).unsqueeze(0)  # [1, H*W, D]


# -------------------------
# CrossAttention
# -------------------------
class CrossAttention(nn.Module):
    def __init__(self, q_dim=512, kv_dim=512, out_dim=512, num_heads=8,
                 qkv_bias=False, qk_norm=True, attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm):
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

# -------------------------
# Encoder Block: CrossAttention + MLP
# -------------------------
class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CrossAttention(
            q_dim=dim, kv_dim=dim, out_dim=dim,
            num_heads=num_heads, qkv_bias=True, qk_norm=True
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x, kv):
        x = x + self.attn(self.norm1(x), kv)
        x = x + self.mlp(self.norm2(x))
        return x

# -------------------------
# EncoderMambaV2
# -------------------------
class EncoderMamba(nn.Module):
    def __init__(self, in_channels=[256, 512, 1024], target_dim=None, mlp_ratio=4. ,depth=4):
        super().__init__()
        if target_dim is None:
            target_dim = in_channels[1]
        self.low_adjust = nn.Sequential(
            BaseConv(in_channels[0], in_channels[0], 3, 1),
            BaseConv(in_channels[0], target_dim, 3, 2)
        )
        self.mid_adjust = BaseConv(in_channels[1], target_dim, 1, 1)
        self.high_adjust = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            BaseConv(in_channels[2], target_dim, 1, 1)
        )
        self.blocks = nn.ModuleList([
            EncoderBlock(dim=target_dim, num_heads=8, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

    def forward(self, features):
        low, mid, high = features  # [B, L, C, H, W]
        B, L, _, H, W = mid.shape

        def reshape_and_adjust(feat, adjust):
            feat = feat.view(B * L, *feat.shape[2:])         # [B*L, C, H, W]
            feat = adjust(feat)                               # [B*L, D, H, W]
            return rearrange(feat, 'bl c h w -> bl (h w) c')  # [B*L, HW, D]

        low_seq = reshape_and_adjust(low, self.low_adjust)
        mid_seq = reshape_and_adjust(mid, self.mid_adjust)
        high_seq = reshape_and_adjust(high, self.high_adjust)

        _, N, C = mid_seq.shape
        pos = get_2d_sincos_pos_embed(C // 2, H, W).to(mid_seq.device)  # [1, HW, C]
        pos = pos.repeat(B * L, 1, 1)

        low_seq += pos
        mid_seq += pos
        high_seq += pos

        kv_input = torch.cat([low_seq, high_seq], dim=1)  # [B*L, 2HW, D]
        x = mid_seq
        for block in self.blocks:
            x = block(x, kv_input)
        return x.view(B, L, H, W, C)
