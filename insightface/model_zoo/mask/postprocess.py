# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# @File: postprocess.py
# @Time: 2022/06/07 14:24:27
# @Author: zhouhl-b
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import sys

import cv2
import numpy as np
# import six
from shapely.geometry import Polygon
from PIL import Image

from .utils.anchor_generator import generate_anchors
from .utils.anchor_decode import decode_bbox
from .utils.nms import single_class_non_max_suppression




def check_and_read_gif(img_path):
    if os.path.basename(img_path)[-3:] in ['gif', 'GIF']:
        gif = cv2.VideoCapture(img_path)
        ret, frame = gif.read()
        if not ret:
            print("Cannot read {}. This gif image maybe corrupted.")
            return None, False
        if len(frame.shape) == 2 or frame.shape[-1] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        imgvalue = frame[:, :, ::-1]
        return imgvalue, True
    return None, False


def create_operators(op_param_list, global_config=None):
    """
    create operators based on the config

    Args:
        params(list): a dict list, used to create some operators
    """
    assert isinstance(
        op_param_list, list), ('operator config should be a list')
    ops = []
    for operator in op_param_list:
        assert isinstance(operator,
                          dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        if global_config is not None:
            param.update(global_config)
        op = eval(op_name)(**param)
        ops.append(op)
    return ops


def draw_text_det_res(dt_boxes, img_path):
    src_im = cv2.imread(img_path)
    for box in dt_boxes:
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        cv2.polylines(src_im, [box], True,
                      color=(255, 255, 0), thickness=2)
    return src_im


class DETPostProcess(object):
    """
    The post process for Face Mask Detection.
    '''
    Main function of detection inference
    :param conf_thresh: the min threshold of classification probabity.
    :param iou_thresh: the IOU threshold of NMS
    :param draw_result: whether to daw bounding box to the image.
    :param show_result: whether to display the image.
    :return:
    '''
    """

    def __init__(self,
                conf_thresh=0.5,
                iou_thresh=0.4,
                max_candidates=1000,
                **kwargs):
        # id2class = {0: 'Mask', 1: 'NoMask'}
        feature_map_sizes = [[45, 45], [23, 23], [12, 12], [6, 6], [4, 4]]
        anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
        anchor_ratios = [[1, 0.62, 0.42]] * 5

        self.feature_map_sizes = feature_map_sizes
        self.anchor_sizes = anchor_sizes
        self.anchor_ratios = anchor_ratios

        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.max_candidates = max_candidates
        print(self.conf_thresh, self.iou_thresh)

    def make_anchors(self, feature_map_sizes, anchor_sizes, anchor_ratios):
        # generate anchors
        anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

        # for inference , the batch size is 1, the model output shape is [1, N, 4],
        # so we expand dim for anchors to [1, anchor_num, 4]
        anchors_exp = np.expand_dims(anchors, axis=0)
        return anchors_exp

    def __call__(self, img, y_bboxes_output, y_cls_output):
        anchors_exp = self.make_anchors(self.feature_map_sizes, self.anchor_sizes, self.anchor_ratios)
        y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
        y_cls = y_cls_output[0]
        # To speed up, do single class NMS, not multiple classes NMS.
        bbox_max_scores = np.max(y_cls, axis=1)
        bbox_max_score_classes = np.argmax(y_cls, axis=1)

        height, width, _ = img.shape
        # image = img.copy()
        
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        # result_str = ''
        # right_num = 0
        res = []
        output_info = []

        # keep_idx is the alive bounding box after nms.
        keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                    bbox_max_scores,
                                                    conf_thresh=self.conf_thresh,
                                                    iou_thresh=self.iou_thresh,
                                                    )
        # print(len(keep_idxs))
        for idx in keep_idxs:
            conf = float(bbox_max_scores[idx])
            # print(conf)
            class_id = bbox_max_score_classes[idx]
            # print(class_id)
            bbox = y_bboxes[idx]
            # clip the coordinate, avoid the value exceed the image boundary.
            xmin = max(0, int(bbox[0] * width))
            ymin = max(0, int(bbox[1] * height))
            xmax = min(int(bbox[2] * width), width)
            ymax = min(int(bbox[3] * height), height)

    
            output_info.append([class_id, conf, xmin, ymin, xmax, ymax])
            # res.append([class_id, conf])
            
            # result_str += '{}\t{}\t{}\t{}\n'.format(img_idx, imgPath, conf, class_id)
            # wf.write(result_str)
        

        return output_info
