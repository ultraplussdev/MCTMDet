#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
"""
Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
"""

import math
import random

import cv2
import numpy as np

from yolox.utils import xyxy2cxcywh


def augment_hsv(img, hgain=5, sgain=30, vgain=30):
    hsv_augs = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]  # random gains
    hsv_augs *= np.random.randint(0, 2, 3)  # random selection of h, s, v
    hsv_augs = hsv_augs.astype(np.int16)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

    img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)

    cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def get_aug_params(value, center=0):
    if isinstance(value, float):
        return random.uniform(center - value, center + value)
    elif len(value) == 2:
        return random.uniform(value[0], value[1])
    else:
        raise ValueError(
            "Affine params should be either a sequence containing two values\
             or single float values. Got {}".format(value)
        )


def get_affine_matrix(
    target_size,
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    twidth, theight = target_size

    # Rotation and Scale
    angle = get_aug_params(degrees)
    scale = get_aug_params(scales, center=1.0)

    if scale <= 0.0:
        raise ValueError("Argument scale should be positive")

    R = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)

    M = np.ones([2, 3])
    # Shear
    shear_x = math.tan(get_aug_params(shear) * math.pi / 180)
    shear_y = math.tan(get_aug_params(shear) * math.pi / 180)

    M[0] = R[0] + shear_y * R[1]
    M[1] = R[1] + shear_x * R[0]

    # Translation
    translation_x = get_aug_params(translate) * twidth  # x translation (pixels)
    translation_y = get_aug_params(translate) * theight  # y translation (pixels)

    M[0, 2] = translation_x
    M[1, 2] = translation_y

    return M, scale


def apply_affine_to_bboxes(targets, target_size, M, scale):
    num_gts = len(targets)

    # warp corner points
    twidth, theight = target_size
    corner_points = np.ones((4 * num_gts, 3))
    corner_points[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
        4 * num_gts, 2
    )  # x1y1, x2y2, x1y2, x2y1
    corner_points = corner_points @ M.T  # apply affine transform
    corner_points = corner_points.reshape(num_gts, 8)

    # create new boxes
    corner_xs = corner_points[:, 0::2]
    corner_ys = corner_points[:, 1::2]
    new_bboxes = (
        np.concatenate(
            (corner_xs.min(1), corner_ys.min(1), corner_xs.max(1), corner_ys.max(1))
        )
        .reshape(4, num_gts)
        .T
    )

    # clip boxes
    new_bboxes[:, 0::2] = new_bboxes[:, 0::2].clip(0, twidth)
    new_bboxes[:, 1::2] = new_bboxes[:, 1::2].clip(0, theight)

    targets[:, :4] = new_bboxes

    return targets


def random_affine(
    img,
    targets=(),
    target_size=(640, 640),
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    M, scale = get_affine_matrix(target_size, degrees, translate, scales, shear)

    img = cv2.warpAffine(img, M, dsize=target_size, borderValue=(114, 114, 114))

    # Transform label coordinates
    if len(targets) > 0:
        targets = apply_affine_to_bboxes(targets, target_size, M, scale)

    return img, targets


def _mirror(image, boxes, prob=0.5):
    _, width, _ = image.shape
    if random.random() < prob:
        image = image[:, ::-1]
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes


def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


class TrainTransform:
    def __init__(self, max_labels=50, flip_prob=0.5, hsv_prob=1.0):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob

    def __call__(self, image, targets, input_dim):
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()
        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, 5), dtype=np.float32)
            image, r_o = preproc(image, input_dim)
            return image, targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :4]
        labels_o = targets_o[:, 4]
        # bbox_o: [xyxy] to [c_x,c_y,w,h]
        boxes_o = xyxy2cxcywh(boxes_o)

        if random.random() < self.hsv_prob:
            augment_hsv(image)
        image_t, boxes = _mirror(image, boxes, self.flip_prob)
        height, width, _ = image_t.shape
        image_t, r_ = preproc(image_t, input_dim)
        # boxes [xyxy] 2 [cx,cy,w,h]
        boxes = xyxy2cxcywh(boxes)
        boxes *= r_

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]

        if len(boxes_t) == 0:
            image_t, r_o = preproc(image_o, input_dim)
            boxes_o *= r_o
            boxes_t = boxes_o
            labels_t = labels_o

        labels_t = np.expand_dims(labels_t, 1)

        targets_t = np.hstack((labels_t, boxes_t))
        padded_labels = np.zeros((self.max_labels, 5))
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
            : self.max_labels
        ]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        return image_t, padded_labels


class ValTransform:
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, swap=(2, 0, 1), legacy=False):
        self.swap = swap
        self.legacy = legacy

    # assume input is cv2 img for now
    def __call__(self, img, res, input_size):
        img, _ = preproc(img, input_size, self.swap)
        if self.legacy:
            img = img[::-1, :, :].copy()
            img /= 255.0
            img -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            img /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        return img, np.zeros((1, 5))

class TrainTransformMamba:
    def __init__(self, max_labels=50, flip_prob=0.5, hsv_prob=1.0):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob

    def __call__(self, images, targets, input_dim):
        assert len(images.shape)==4
        B = images.shape[0]
        has_boxes = any(t is not None and len(t) > 0 for t in targets)
        if not has_boxes:
            images_t, r_o = preproc_batch(images, input_dim)
            targets_t = np.zeros((B, self.max_labels, 5), dtype=np.float32)
            return images_t, targets_t

        image_o = images.copy()
        targets_o = [t.copy() if t is not None and len(t) > 0 else np.zeros((0, 5), dtype=np.float32) for t in targets]
        _, height_o, width_o, _ = image_o.shape

        boxes_o = [t[:, :4].copy() if t.shape[0] > 0 else np.zeros((0, 4), dtype=np.float32) for t in targets_o]
        labels_o = [t[:, 4].copy() if t.shape[0] > 0 else np.array([], dtype=np.float32) for t in targets_o]

        boxes_o = xyxy2cxcywh_batch(boxes_o)  # [B], each [N_b, 4] in (cx, cy, w, h)
        images_t = images.copy()
        if random.random() < self.hsv_prob:
            augment_hsv_batch(images_t)
        images_t, boxes = _mirror_batch(images_t, boxes_o, self.flip_prob)
        _, height, width, _ = images_t.shape
        images_t, r_ = preproc_batch(images_t, input_dim)
        # boxes [xyxy] 2 [cx,cy,w,h]
        targets_t = np.zeros((B, self.max_labels, 5), dtype=np.float32)
        for b in range(B):
            if boxes[b] is None or len(boxes[b]) == 0:
                continue

            # 过滤小框
            mask_b = np.minimum(boxes[b][:, 2], boxes[b][:, 3]) > 1
            boxes_t = boxes[b][mask_b]
            labels_t = labels_o[b][mask_b]
            if len(boxes_t) == 0:
                image_t_b, r_o = preproc(image_o[b], input_dim)
                boxes_t = boxes_o[b] * r_o
                labels_t = labels_o[b]
                images_t[b] = image_t_b

            # 填充目标
            labels_t = np.expand_dims(labels_t, 1)
            targets_t_b = np.hstack((labels_t, boxes_t))
            targets_t[b, range(len(targets_t_b))[:self.max_labels]] = targets_t_b[:self.max_labels]

        images_t = np.ascontiguousarray(images_t, dtype=np.float32)
        targets_t = np.ascontiguousarray(targets_t, dtype=np.float32)
        return images_t, targets_t

class ValTransformMamba:
    def __init__(self, swap=(2, 0, 1), legacy=False, max_labels=50):
        self.swap = swap
        self.legacy = legacy
        self.max_labels = max_labels
    def __call__(self, images, targets_list, input_size):
        """
        Args:
            images: np.ndarray, [B, H, W, C], batch of images
            targets_list: list, [B], each element is np.ndarray [N_b, 5] or None
            input_size: tuple, target image size (e.g., (416, 416))
        Returns:
            images_t: np.ndarray, [B, C, H_new, W_new], processed images
            targets_t: np.ndarray, [B, max_labels, 5], padded targets
        """

        if len(images.shape) == 3:
            images = images[np.newaxis, ...]
            targets_list = [targets_list] if targets_list is not None else [None]

        B = images.shape[0]

        images_t, r_ = preproc_batch(images, input_size, self.swap)
        if self.legacy:
            images_t = images_t[:, ::-1, :, :].copy()
            images_t = images_t / 255.0
            mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
            images_t = (images_t - mean) / std


        targets_t = np.zeros((B, self.max_labels, 5), dtype=np.float32)
        for b in range(B):
            if targets_list[b] is None or len(targets_list[b]) == 0:
                continue

            boxes = targets_list[b][:, :4].copy()
            labels = targets_list[b][:, 4].copy()
            boxes *= r_

            targets_t_b = np.hstack((labels[:, np.newaxis], boxes))
            targets_t[b, :min(len(targets_t_b), self.max_labels)] = targets_t_b[:self.max_labels]

        return images_t, targets_t

def preproc_batch(imgs, input_size, swap=(2, 0, 1)):
    """
    Preprocess a batch of images.

    Args:
        imgs: np.ndarray, [B, H, W, C] or [H, W, C], input images
        input_size: tuple, target size (height, width), e.g., (416, 416)
        swap: tuple, channel swap order, e.g., (2, 0, 1) for [H, W, C] -> [C, H, W]

    Returns:
        padded_img: np.ndarray, [B, C, H_new, W_new], processed images
        r: float or np.ndarray, scaling ratio(s)
    """

    if len(imgs.shape) == 3:
        imgs = imgs[np.newaxis, ...]  # [H, W, C] -> [1, H, W, C]

    B, H, W, C = imgs.shape


    padded_img = np.ones((B, input_size[0], input_size[1], 3), dtype=np.uint8) * 114

    r = min(input_size[0] / H, input_size[1] / W)


    resized_imgs = np.array([
        cv2.resize(
            imgs[b],
            (int(W * r), int(H * r)),
            interpolation=cv2.INTER_LINEAR
        ) for b in range(B)
    ]).astype(np.uint8)  # [B, H*r, W*r, C]

    padded_img[:, :int(H * r), :int(W * r), :] = resized_imgs
    padded_img = padded_img.transpose(0, 3, 1, 2)  # [B, H_new, W_new, C] -> [B, C, H_new, W_new]

    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)

    return padded_img, r

def _mirror_batch(images, boxes, prob=0.5):
    """
    Apply horizontal mirroring to a batch of images and their bboxes.

    Args:
        images: np.ndarray, [B, H, W, C] or [H, W, C], input images
        boxes: list, [B], each element is np.ndarray [N_b, 4] in (x1, y1, x2, y2) or (cx, cy, w, h)
        prob: float, probability of mirroring

    Returns:
        images: np.ndarray, [B, H, W, C] or [H, W, C], mirrored images
        boxes: list, [B], each element is np.ndarray [N_b, 4], adjusted bboxes
    """
    if len(images.shape) == 3:
        images = images[np.newaxis, ...]
        boxes = [boxes] if boxes is not None else [None]

    B, H, W, C = images.shape

    if random.random() < prob:

        images = images[:, :, ::-1, :]
        for b in range(B):
            if boxes[b] is not None and len(boxes[b]) > 0:
                boxes[b] = boxes[b].copy()
                boxes[b][:, 0::2] = W - boxes[b][:, 2::-2]

    return images, boxes

def augment_hsv_batch(img, hgain=5, sgain=30, vgain=30):
    """
    Apply HSV augmentation to a batch of images in-place.

    Args:
        img: np.ndarray, [B, H, W, C] or [H, W, C], input images in BGR format
        hgain, sgain, vgain: float, gain ranges for H, S, V channels
    """
    B, H, W, C = img.shape
    hsv_augs = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]  # [H, S, V]
    hsv_augs *= np.random.randint(0, 2, 3)
    hsv_augs = hsv_augs.astype(np.int16)

    img_hsv = np.stack([cv2.cvtColor(img[b], cv2.COLOR_BGR2HSV).astype(np.int16) for b in range(B)], axis=0)

    img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180  # Hue
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)  # Saturation
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)  # Value


    for b in range(B):
        cv2.cvtColor(img_hsv[b].astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img[b])

def xyxy2cxcywh_batch(boxes_o):
    if isinstance(boxes_o, np.ndarray) and len(boxes_o.shape) == 2:
        boxes_o = [boxes_o]
    boxes_out = []
    for bboxes in boxes_o:
        if bboxes is None or len(bboxes) == 0:
            boxes_out.append(np.zeros((0, 4), dtype=bboxes.dtype if bboxes is not None else np.float32))
            continue
        bboxes = bboxes.copy()
        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
        bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
        bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
        bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
        boxes_out.append(bboxes)
    if len(boxes_out) == 1:
        return boxes_out[0]
    return boxes_out