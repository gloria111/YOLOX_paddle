#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

# import torch
import paddle

# import torch.nn as nn
import paddle.nn as nn
import numpy


class IOUloss(nn.Layer):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        pred_shape=numpy.array(pred).shape[0]
        target_shape = numpy.array(target).shape[0]

        assert pred_shape == target_shape
        if pred_shape !=0:
            pred = pred.reshape([-1, 4])
            target = target.reshape([-1, 4])
        tl = paddle.maximum(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = paddle.minimum(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = paddle.prod(pred[:, 2:], 1)
        area_g = paddle.prod(target[:, 2:], 1)
        if pred_shape !=0:
            en = (tl < br).astype(tl.dtype).prod(axis=1)
        else:
            en = tl.prod(axis=1)
        area_i = paddle.prod(br - tl, 1) * en
        iou = (area_i) / (area_p + area_g - area_i + 1e-16)

        if self.loss_type == "iou":
            if pred_shape !=0:
                loss = 1 - iou ** 2
            else:
                loss = 1- iou
        elif self.loss_type == "giou":
            c_tl = paddle.minimum(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = paddle.maximum(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = paddle.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_i) / area_c.clamp(1e-16)
            loss = 1 - giou.clip(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
