#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import numpy as np
# import paddle.fluid as fluid
from .ops import multiclass_nms

# import torch
import paddle
# import torchvision
import sys
# sys.path.append(r'/home/aistudio/YOLOX/yolox/utils/nms/')
# from nms import multiclass_nms
__all__ = [
    "filter_box",
    "postprocess",
    "bboxes_iou",
    "matrix_iou",
    "adjust_box_anns",
    "xyxy2xywh",
    "xyxy2cxcywh",
]


def filter_box(output, scale_range):
    """
    output: (N, 5+class) shape
    """
    min_scale, max_scale = scale_range
    w = output[:, 2] - output[:, 0]
    h = output[:, 3] - output[:, 1]
    keep = (w * h > min_scale * min_scale) & (w * h < max_scale * max_scale)
    return output[keep]

#############################################################################################
# def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
#     # print(".............boxes中出现分数过低，只判定第一类类别的物体，试图将score_threshold设置为0.05.....after nms............")

#     box_corner = paddle.ones_like(prediction)

#     box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
#     box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
#     box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
#     box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
#     prediction[:, :, :4] = box_corner[:, :, :4]

#     output = [None for _ in range(len(prediction))]
#     for i, image_pred in enumerate(prediction):#prediction.shape[64, 8400, 85]
      
#         # If none are remaining => process next image
#         if not image_pred.shape(0):
#             continue
#         # Get score and class with highest confidence

#         class_conf = paddle.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
#         class_pred = paddle.argmax(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)#class_pred==0([8400]个0)
#         conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= 0.005).squeeze()


#         # print("boxes......................img_pred.....")
#         # print(image_pred[:, 4] * class_conf.squeeze()>= 0.7)

        # # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        # class_pred = paddle.cast(class_pred, 'float32')
        # detections = paddle.concat((image_pred[:, :5], class_conf, class_pred), 1)

#         # print("boxes.......................detections")
#         # print(detections)
#         # print("boxes.......................mask")
#         # print(conf_mask)

        
        # # detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        # # conf_mask = paddle.expand(conf_mask, shape=[7, 8400])
        # temp_shape = detections.shape[1]# detections shape = [64, 8400, 85]
        # mask = paddle.reshape(conf_mask,[-1,1])
        # conf_mask = paddle.concat([mask,mask,mask,mask,mask,mask,mask], axis=1)#conf_mask.shape=detections.shape=[8400,7]
        # detections = paddle.masked_select(detections,conf_mask)
        # col = detections.shape[0]/temp_shape
        # detections = paddle.reshape(detections,[-1,temp_shape])#[5890, 7]
        

#         scores = detections[:, 4] * detections[:, 5] 

#         # scores = paddle.unsqueeze(scores, axis=0)
#         # ones = paddle.ones(shape=[80, 1])
#         # scores = paddle.matmul(ones, scores)
#         # scores = paddle.unsqueeze(scores, axis=0)
#         # print(detections[:, 6] )
#         # for j in range(0,8399):
#         #     a1 = detections[:, 6][0]
#         #     a1 = paddle.cast(a1, 'int32')
#         #     scores[a1][j] = scores_1d[j]
#         # scores = paddle.reshape(scores,[-1,1])
#         # scores = paddle.concat((output[i], detections))
#         #上面这段应该是为了对齐multiclass_nms的api接口写的调整代码

#         scores = paddle.unsqueeze(scores, axis=0)
#         scores = paddle.unsqueeze(scores, axis=0)

#         # nms_cfg = dict(
#         #     score_thr=0.05,
#         #     nms=dict(type='nms',iou_threshold=0.6),
#         #     )
#         bboxes = detections[:, :4]

#         bboxes = paddle.unsqueeze(bboxes, axis=0)

#         if not detections.shape[0]:
#             continue
# ###################################################################################################
def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    print("pre..........................................................................................")
    print(prediction[:,:,6]>0.3)
    # box_corner = prediction.new(prediction.shape)
    box_corner = paddle.ones_like(prediction)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.shape[0]:
            continue

        # Get score and class with highest confidence
        # class_conf, class_pred = paddle.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
        class_conf = paddle.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
        class_pred = paddle.argmax(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)#class_pred==0([8400]个0)
        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= 0.005).squeeze()
        class_pred = paddle.cast(class_pred, 'float32')
        detections = paddle.concat((image_pred[:, :5], class_conf, class_pred), 1)
        temp_shape = detections.shape[1]# detections shape = [64, 8400, 85]
        mask = paddle.reshape(conf_mask,[-1,1])
        conf_mask = paddle.concat([mask,mask,mask,mask,mask,mask,mask], axis=1)#conf_mask.shape=detections.shape=[8400,7]
        detections = paddle.masked_select(detections,conf_mask)
        col = detections.shape[0]/temp_shape
        detections = paddle.reshape(detections,[-1,temp_shape])#[5890, 7]
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        # detections = paddle.concat((image_pred[:, :5], class_conf, class_pred), 1)
        # detections = paddle.concat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        # detections = detections[conf_mask]
                
        boxes = detections[:, :4]
        scores = detections[:, 4] * detections[:, 5]
        scores = paddle.unsqueeze(scores, axis=0)
        scores = paddle.unsqueeze(scores, axis=0)
        boxes = paddle.unsqueeze(boxes, axis=0)

        if not detections.shape[0]:
            continue

        if class_agnostic:
                nms_out = multiclass_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        else:  
            nms_out_index = multiclass_nms(
                # paddle.unsqueeze(detections[:, :4], axis=0, name=None),
                # paddle.unsqueeze(detections[:, 4] * detections[:, 5], axis=1, name=None),
                boxes,
                scores,
                score_threshold=0.03,
                nms_top_k=100,
                keep_top_k=100,
                nms_threshold=nms_thre,
                return_index=True,
            )  
            # nms_out = multiclass_nms(
            #     bboxes,
            #     scores,
            #     labels=class_pred,
            #     num_classes = num_classes,
            #     score_thr = 0.5/10,
            #     nms_cfg = nms_thre,
            #     max_num=100,
            #     # nms_threshold=nms_thre,
            #     # keep_top_k=1000,
            # )
        detections = detections[nms_out_index[2]]

        if output[i] is None:
            output[i] = detections
        else:
            output[i] = paddle.concat((output[i], detections))

    return paddle.to_tensor(output)


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = paddle.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = paddle.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = paddle.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = paddle.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = paddle.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = paddle.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = paddle.prod(bboxes_a[:, 2:], 1)
        area_b = paddle.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = paddle.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i + 1e-12)


def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    return bbox


def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes


def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes
