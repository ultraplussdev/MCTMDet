#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import os
import random

import torch
import torch.distributed as dist
import torch.nn as nn
from yolox.data.datasets import vid
from yolox.exp.base_exp import BaseExp
from yolox.data.data_augment import ValTransformMamba
from loguru import logger

class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        self.archi_name = 'MCTMDet'
        self.backbone_name = 'MCSP'
        # ---------------- model config ---------------- #
        # detect classes number of model
        self.num_classes = 30
        # factor of model depth
        self.depth = 1.00
        # factor of model width
        self.width = 1.00
        # activation name. For example, if using "relu", then "silu" will be replaced to "relu".
        self.act = "silu"



        # ---------------- dataloader config ---------------- #
        # set worker to 12 for shorter dataloader init time
        # If your training process cost many memory, reduce this value.
        self.data_num_workers = 12
        self.input_size = (640, 640)  # (height, width)
        # Actual multiscale ranges: [640 - 5 * 32, 640 + 5 * 32].
        # To disable multiscale training, set the value to 0.
        self.multiscale_range = 5
        # You can uncomment this line to specify a multiscale range
        # self.random_size = (14, 26)
        # dir of dataset images, if data_dir is None, this project will use `datasets` dir
        self.data_dir = '/root/autodl-tmp/datasets/datasets'
        # name of annotation file for training
        self.vid_train_path = './yolox/data/datasets/train_seq.npy'  #'./yolox/data/datasets/train_seq.npy'
        self.vid_val_path = './yolox/data/datasets/val_seq.npy'
        # path to vid name list

        # --------------- transform config ----------------- #
        # prob of applying mosaic aug
        self.mosaic_prob = 1.0
        # prob of applying mixup aug
        self.mixup_prob = 1.0
        # prob of applying hsv aug
        self.hsv_prob = 1.0
        # prob of applying flip aug
        self.flip_prob = 0.5
        # rotation angle range, for example, if set to 2, the true range is (-2, 2)
        self.degrees = 10.0
        # translate range, for example, if set to 0.1, the true range is (-0.1, 0.1)
        self.translate = 0.1
        self.mosaic_scale = (0.1, 2)
        # apply mixup aug or not
        self.enable_mixup = True
        self.mixup_scale = (0.5, 1.5)
        # shear angle range, for example, if set to 2, the true range is (-2, 2)
        self.shear = 2.0

        # --------------  training config --------------------- #

        # epoch number used for warmup
        self.warmup_epochs = 1
        # max training epoch
        self.max_epoch = 13
        # minimum learning rate during warmup
        self.warmup_lr = 0
        self.min_lr_ratio = 0.1
        # learning rate for one image. During training, lr will multiply batchsize.
        self.basic_lr_per_img = 0.002 / 64.0
        # name of LRScheduler
        self.scheduler = "yoloxwarmcos"
        # last #epoch to close augmention like mosaic
        self.no_aug_epochs = 2
        # apply EMA during training
        self.ema = True

        # weight decay of optimizer
        self.weight_decay = 5e-4
        # momentum of optimizer
        self.momentum = 0.9
        # log period in iter, for example,
        # if set to 1, user could see log every iteration.
        self.print_interval = 10
        # eval period in epoch, for example,
        # if set to 1, model will be evaluate after every epoch.
        self.eval_interval = 1
        # save history checkpoint or not.
        # If set to False, yolox will only save latest and best ckpt.
        self.save_history_ckpt = True
        # name of experiment
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # -----------------  testing config ------------------ #
        # output image size during evaluation/test
        self.test_size = (640, 640)
        # confidence threshold during evaluation/test,
        # boxes whose scores are less than test_conf will be filtered
        self.test_conf = 0.001
        # nms threshold
        self.nmsthre = 0.5

    def get_model(self):
        from yolox.models import YOLOXHead, MambaYOLOPAFPN, CSPDarknet,MambaYOLOX,EncoderMamba

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            class MambaDarknet(nn.Module):
                def __init__(self,exp, depth, width):
                    super().__init__()
                    self.darknet = CSPDarknet(dep_mul=depth, wid_mul=width, out_features=("dark3", "dark4", "dark5"))
                    base_channels = [int(256 * width), int(512 * width), int(1024 * width)]
                    self.encoder = EncoderMamba(base_channels)
                    self.out_features = ("dark3", "dark4", "dark5")
                def forward(self, frames):
                    if frames.dim()==4:
                        frames = frames.unsqueeze(1)
                    B,L,C,H,W=frames.shape
                    frames=frames.view(B * L, C, H, W)
                    features = self.darknet(frames)
                    features = [features[out_f].view(B, L, *features[out_f].shape[1:])
                                     for out_f in self.out_features]
                    context = self.encoder(features)
                    return features,context
            backbone = MambaDarknet(self,
                self.depth, self.width)
            neck = MambaYOLOPAFPN(depth=self.depth, width=self.width, in_features=("dark3", "dark4", "dark5"),
                                   act=self.act)
            head = YOLOXHead(self.num_classes, self.width, act=self.act)
            self.model = MambaYOLOX(backbone=backbone, neck=neck, head=head,base_channels=self.base_channels,memory_enabled=self.memory_enabled)
            print(self.model)

            for param in backbone.darknet.parameters():
                param.requires_grad = False


            def fix_bn(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.model.apply(init_yolo)
            if hasattr(self, 'fix_bn') and self.fix_bn:
                self.model.apply(fix_bn)
            self.model.head.initialize_biases(1e-2)
            return self.model



    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size


            pg0, pg1, pg2 = [], [], []  # BN weights, weights with decay, biases

            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if isinstance(param, nn.BatchNorm2d) or "bn" in name:
                    pg0.append(param)  # no decay
                elif "bias" in name:
                    pg2.append(param)  # biases
                else:
                    pg1.append(param)  # weights with decay
                    logger.info(f"Optimizing parameter: {name}")

            optimizer = torch.optim.SGD(
                pg0, lr=lr, momentum=self.momentum, nesterov=True
            )
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )
            optimizer.add_param_group({"params": pg2})

            self.optimizer = optimizer

        return self.optimizer

    def get_data_loader(
            self, batch_size, is_distributed, no_aug=False, cache_img=False
    ):
        from yolox.data import TrainTransformMamba, TrainTransform
        from yolox.data.datasets.mosaicdetection import MosaicDetection
        dataset = vid.VIDDataset_mamba(file_path=self.vid_train_path,
                                 img_size=self.input_size,
                                 preproc=TrainTransformMamba(
                                     max_labels=50,
                                     flip_prob=self.flip_prob,
                                     hsv_prob=self.hsv_prob),
                                 lframe=self.lframe,
                                 gframe=self.gframe,
                                 dataset_pth=self.data_dir,
                                 local_stride=self.local_stride,
                                 )
        if self.use_aug:
            # NO strong aug by defualt
            dataset = MosaicDetection(
                dataset=dataset,
                mosaic=False,
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=120,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob),
                degrees=self.degrees,
                translate=self.translate,
                mosaic_scale=self.mosaic_scale,
                mixup_scale=self.mixup_scale,
                shear=self.shear,
                enable_mixup=self.enable_mixup,
                mosaic_prob=self.mosaic_prob,
                mixup_prob=self.mixup_prob,
            )
        dataset = vid.get_trans_loader_mamba(batch_size=batch_size, data_num_workers=4, dataset=dataset)
        return dataset
    def get_eval_loader(self, batch_size, tnum=None, data_num_workers=4,formal=False):
        if tnum == None:
            tnum = self.tnum
        from yolox.data import ValTransformMamba
        dataset_val = vid.VIDDataset_mamba(file_path=self.vid_val_path,
                                     img_size=self.test_size, preproc=ValTransformMamba(), lframe=self.lframe_val,
                                     gframe=self.gframe_val, val=True, dataset_pth=self.data_dir, tnum=tnum,formal=formal,
                                     traj_linking=self.traj_linking, local_stride=self.local_stride,)
        val_loader = vid.vid_val_loader_mamba(batch_size=batch_size,
                                        data_num_workers=data_num_workers,
                                        dataset=dataset_val, )

        return val_loader
    def get_lr_scheduler(self, lr, iters_per_epoch):
        from yolox.utils import LRScheduler

        scheduler = LRScheduler(
            self.scheduler,
            lr,
            iters_per_epoch,
            self.max_epoch,
            warmup_epochs=self.warmup_epochs,
            warmup_lr_start=self.warmup_lr,
            no_aug_epochs=self.no_aug_epochs,
            min_lr_ratio=self.min_lr_ratio,
        )
        return scheduler
    def get_evaluator(self, val_loader):
        from yolox.evaluators.vid_evaluator_v2_mamba import VIDEvaluator

        # val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = VIDEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            lframe=self.lframe_val,
            gframe=self.gframe_val,
            first_only = False,
        )
        return evaluator

    def get_trainer(self, args):
        from yolox.core import Trainer
        trainer = Trainer(self, args)
        # NOTE: trainer shouldn't be an attribute of exp object
        return trainer

    def eval(self, model, evaluator, is_distributed, half=False):
        return evaluator.evaluate(model, is_distributed, half)
