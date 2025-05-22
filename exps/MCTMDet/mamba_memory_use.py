import os
import torch.nn as nn
import sys
import torch
from skimage.filters.rank import enhance_contrast

sys.path.append("..")
from exps.exp_base.exp_base import Exp as MyExp
from yolox.data.datasets import vid
from yolox.utils import postprocess
from loguru import logger
class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33  # 1#0.67
        self.width = 0.5  # 1#0.75
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.max_epoch = 13
        # Define yourself dataset path

        self.warmup_epochs = 0
        self.no_aug_epochs = 2
        self.pre_no_aug = 2
        self.eval_interval = 1
        self.lframe = 4
        self.lframe_val = 4
        self.gframe = 0
        self.gframe_val = 0
        self.tnum = -1
        self.output_dir = "./MCTMDet_outputs"
        self.use_aug = False
        self.use_aggregation = True
        self.memory_enabled = False
        self.memory_bank = {}  # {cls_id: [context]}
        self.base_channels = [int(256 * self.width), int(512 * self.width), int(1024 * self.width)]
        self.traj_linking=False
        self.local_stride = 1
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
