#copy from https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/nms.py
import os

import numpy as np
import paddle
import importlib
from collections import namedtuple
# This function is modified from: https://github.com/pytorch/vision/
# def load_ext(name, funcs):
#         ext = importlib.import_module('mmcv.' + name)
#         for fun in funcs:
#             assert hasattr(ext, fun), f'{fun} miss in module {name}'
#         return ext


# ext_module = load_ext(
#     '_ext', ['nms', 'softnms', 'nms_match', 'nms_rotated'])

# def nms(boxes, scores, iou_threshold, offset=0, score_threshold=0, max_num=-1):
#     assert isinstance(boxes, (paddle.Tensor, np.ndarray))
#     assert isinstance(scores, (paddle.Tensor, np.ndarray))
#     is_numpy = False
#     if isinstance(boxes, np.ndarray):
#         is_numpy = True
#         boxes = paddle.to_tensor(boxes)
#     if isinstance(scores, np.ndarray):
#         scores = paddle.to_tensor(scores)
#     assert boxes.size(1) == 4
#     assert boxes.size(0) == scores.size(0)
#     assert offset in (0, 1)
#     if torch.__version__ == 'parrots':
#         indata_list = [boxes, scores]
#         indata_dict = {
#             'iou_threshold': float(iou_threshold),
#             'offset': int(offset)
#         }
#         inds = ext_module.nms(*indata_list, **indata_dict)
#     else:
#         inds = NMSop.apply(boxes, scores, iou_threshold, offset,
#                            score_threshold, max_num)
   
#     dets = paddle.concat((boxes[inds], scores[inds].reshape(-1, 1)), dim=1)
#     if is_numpy:
#         dets = dets.cpu().numpy()
#         inds = inds.cpu().numpy()
#     return dets, inds


def batched_nms(boxes, scores, idxs, nms_cfg, class_agnostic=False):

    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop('class_agnostic', class_agnostic)
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + paddle.to_tensor(1).to(boxes))
        boxes_for_nms = boxes + offsets[:, None]

    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = eval(nms_type)

    split_thr = nms_cfg_.pop('split_thr', 10000)
    # Won't split to multiple nms nodes when exporting to onnx

    max_num = nms_cfg_.pop('max_num', -1)
    total_mask = scores.new_zeros(scores.size(), dtype='bool')
    # Some type of nms would reweight the score, such as SoftNMS
    scores_after_nms = scores.new_zeros(scores.size())
    for id in paddle.unique(idxs):
        mask = (idxs == id).nonzero(as_tuple=False).view(-1)
        dets, keep = nms_op(boxes_for_nms[mask], scores[mask], **nms_cfg_)
        total_mask[mask[keep]] = True
        scores_after_nms[mask[keep]] = dets[:, -1]
    keep = total_mask.nonzero(as_tuple=False).view(-1)

    scores, inds = scores_after_nms[keep].sort(descending=True)
    keep = keep[inds]
    boxes = boxes[keep]

    if max_num > 0:
        keep = keep[:max_num]
        boxes = boxes[:max_num]
        scores = scores[:max_num]

    return paddle.concat([boxes, scores[:, None]], -1), keep

