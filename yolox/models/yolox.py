#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
import torch
import torch.nn as nn

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
from .mamba_decoder import DecoderMamba
from .memory import MemoryManager
from .mambaTransfomer import MambaTransfomer
from yolox.utils import postprocess
class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_outs, targets, x
            )
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
        else:
            outputs = self.head(fpn_outs)

        return outputs
class CustomYOLOX(nn.Module):
    def __init__(self, backbone=None, neck=None, head=None):
        super().__init__()
        if backbone is None:
            raise ValueError("Backbone cannot be None")
        if neck is None:
            raise ValueError("Neck (FPN) cannot be None")
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, current_frame, local_frames=None, global_frames=None,  targets=None):
        # 单帧模式（推理或 FLOPs 计算）
        if local_frames is None or global_frames is None:
            backbone_outs = self.backbone(current_frame)
            fpn_outs = self.neck(backbone_outs)
            if self.training:
                assert targets is not None
                loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(fpn_outs, targets, current_frame)
                outputs = {
                    "total_loss": loss,
                    "iou_loss": iou_loss,
                    "l1_loss": l1_loss,
                    "conf_loss": conf_loss,
                    "cls_loss": cls_loss,
                    "num_fg": num_fg,
                }
            else:
                outputs = self.head(fpn_outs)
            return outputs

        backbone_outs = self.backbone(current_frame, local_frames, global_frames)
        fpn_outs = self.neck(backbone_outs)
        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(fpn_outs, targets, current_frame)
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
        else:
            outputs = self.head(fpn_outs)
        return outputs
class MambaYOLOX(nn.Module):
    def __init__(self, backbone=None, neck=None, head=None,base_channels=None,memory_enabled=False):
        super().__init__()
        if backbone is None:
            raise ValueError("Backbone cannot be None")
        if neck is None:
            raise ValueError("Neck (FPN) cannot be None")
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.memory_enabled = memory_enabled
        self.mambatransformer = MambaTransfomer(dim=base_channels[1], depth=6, num_heads=8, transformer_blocks=[1,3,5])
        self.decoder = DecoderMamba(in_channels=base_channels, context_dim=base_channels[1])
        self.memory = MemoryManager(ctx_dim=base_channels[1]) if memory_enabled else None
        self.num_classes=30
    def forward(self, frames,  targets=None):
                features, context = self.backbone(frames)
                context_aug = self.mambatransformer(context)
                if self.memory_enabled:
                    with torch.no_grad():
                        self.head.eval()
                        self.neck.eval()
                        context_for_mem = context_aug.detach()
                        coarse_feats = self.decoder(features, context_for_mem)
                        fpn_outs = self.neck(tuple(coarse_feats))
                        coarse_outs = self.head(fpn_outs)
                        detections = postprocess(coarse_outs, self.num_classes)
                        self.head.train(self.training)
                        self.neck.train(self.training)
                    self.memory.update(detections, context_for_mem)
                    memory_ctx = self.memory.retrieve(detections,context_for_mem)
                    context_aug = self.memory.fuse_context(context_aug, memory_ctx)
                    del context_for_mem, coarse_feats, fpn_outs, coarse_outs
                    del detections, memory_ctx
                    torch.cuda.empty_cache()  # 可选，但建议加上
                enhanced_feats = self.decoder(features, context_aug)

                # 5. Neck + Head
                fpn_outs = self.neck(tuple(enhanced_feats))
                if self.training:
                    assert targets is not None
                    loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(fpn_outs, targets, frames)
                    outputs = {
                        "total_loss": loss,
                        "iou_loss": iou_loss,
                        "l1_loss": l1_loss,
                        "conf_loss": conf_loss,
                        "cls_loss": cls_loss,
                        "num_fg": num_fg,
                    }
                else:
                    outputs = self.head(fpn_outs)
                return outputs


