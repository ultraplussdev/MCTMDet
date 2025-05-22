import time

import torch
import torch.nn as nn
import heapq
from collections import defaultdict
from typing import List, Union
from einops import rearrange


class MemoryManager:
    def __init__(self, max_length=2, conf_thre=0.6, use_class_memory=True, ctx_dim=256):
        self.max_length = max_length
        self.conf_thre = conf_thre
        self.use_class_memory = use_class_memory
        self.memory_bank = defaultdict(list) if use_class_memory else []
        self.fuse_conv = nn.Sequential(
            BaseConv(ctx_dim * 2, ctx_dim, ksize=1, stride=1),
            BaseConv(ctx_dim, ctx_dim, ksize=3, stride=1),
        )

    def update(self, detections_list, context_list):
        B, L, H, W, C = context_list.shape
        context_flatten = context_list.view(B * L, H, W, C)

        for i, detections in enumerate(detections_list):
            if detections is None or detections.numel() == 0:
                continue
            frame_ctx = context_flatten[i]
            for det in detections:
                cls_conf = det[5].item()
                cls_id = int(det[6].item())
                if cls_conf < self.conf_thre:
                    continue
                timestamp = time.time()
                if self.use_class_memory:
                    heapq.heappush(self.memory_bank[cls_id], (-cls_conf, timestamp, frame_ctx))
                    if len(self.memory_bank[cls_id]) > self.max_length:
                        heapq.heappop(self.memory_bank[cls_id])
                else:
                    heapq.heappush(self.memory_bank, (-cls_conf, timestamp, frame_ctx))
                    if len(self.memory_bank) > self.max_length:
                        heapq.heappop(self.memory_bank)

    def retrieve(self, detections_list,context_aug):
        if not self.memory_bank:
            return []
        B, L, H, W, C = context_aug.shape
        context_flatten = context_aug.view(B * L, H, W, C)
        memory_contexts = []
        for i,det in enumerate(detections_list):
            if det is None or det.numel() == 0:
                memory_contexts.append(context_flatten[i])
                continue
            for d in det:
                cls_id = int(d[6].item())
                if cls_id in self.memory_bank and len(self.memory_bank[cls_id]) > 0:
                    ctx = self.memory_bank[cls_id][0][2]
                    memory_contexts.append(ctx)
                    break
                else:
                    memory_contexts.append(context_flatten[i])
                    break
        memory_ctx_tensor = torch.stack(memory_contexts)  # [B*L, H, W, C]
        memory_ctx_tensor = memory_ctx_tensor.view(B, L, H, W, C)
        return memory_ctx_tensor

    def fuse_context(self, current_context: torch.Tensor, memory_contexts: List[torch.Tensor], method='concat'):
        self.fuse_conv = self.fuse_conv.to(current_context.device)
        if len(memory_contexts) == 0:
            return current_context

        if method == 'avg':
            return (current_context + memory_contexts) / 2
        elif method == 'concat':
            x = torch.cat([current_context, memory_contexts], dim=-1)  # [B, L, H, W, 2C]
            x = rearrange(x, "b l h w c -> (b l) c h w")
            out = self.fuse_conv(x)
            out = rearrange(out, "(b l) c h w -> b l h w c", b=current_context.shape[0])
            fused = current_context + out
            return fused
        else:
            raise ValueError(f"Unsupported fusion method: {method}")


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