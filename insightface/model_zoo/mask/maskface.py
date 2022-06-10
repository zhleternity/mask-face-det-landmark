# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# @File: maskface.py
# @Time: 2022/06/09 11:10:35
# @Author: zhouhl-b

from __future__ import division
import datetime
import numpy as np
import onnx
import onnxruntime
import os
import os.path as osp
import cv2
import sys
import time

from .postprocess import (DETPostProcess, check_and_read_gif)


def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)

def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i%2] + distance[:, i]
        py = points[:, i%2+1] + distance[:, i+1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)

class MaskFace:
    def __init__(self, model_file=None, session=None):
        self.model_file = model_file
        self.session = session
        self.taskname = 'detection'
        if self.session is None:
            assert self.model_file is not None
            assert osp.exists(self.model_file)
            sess_opt = onnxruntime.SessionOptions()
            self.session = onnxruntime.InferenceSession(self.model_file, sess_opt)

        target_shape = (360, 360)
        id2class = {0: 'Mask', 1: 'NoMask'}
        draw_result=True
        show_result=False
        self.nms_thresh = 0.4
        self.det_thresh = 0.5
        
        self.postprocess_op = DETPostProcess(conf_thresh=0.5,
                                            iou_thresh=0.4,
                                            max_candidates=1000)
        self.image_shape = target_shape
        self.id2class = id2class
        self.draw_result = draw_result
        self.show_result = show_result

        self.use_kps = False
        
        # ONNX
        # sess_opt.log_severity_level = 4
        # sess_opt.enable_cpu_mem_arena = False
        input_cfg = self.session.get_inputs()[0]
        self.input_shape = input_cfg.shape
        self.input_name = self.session.get_inputs()[0].name

        if isinstance(self.input_shape[2], str):
            self.input_size = None
        else:
            self.input_size = tuple(self.input_shape[2:4][::-1])
        
        self.box_output_name = self.session.get_outputs()[0].name
        self.cls_output_name = self.session.get_outputs()[1].name

        self.input_mean = 0.0
        self.input_std = 1.0

    def preprocess_img(self, image, target_shape):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, target_shape)
        image_resized = image_resized.astype(np.float32)
        # image_np = np.ascontiguousarray(image_resized).astype(np.float32) / 255.0 
        image_np = image_resized / 255.0
        image_exp = np.expand_dims(image_np, axis=0)
        image_transposed = image_exp.transpose((0, 3, 1, 2))

        return image_transposed
    
    def prepare(self, ctx_id, **kwargs):
        if ctx_id<0:
            self.session.set_providers(['CPUExecutionProvider'])
        nms_thresh = kwargs.get('nms_thresh', None)
        if nms_thresh is not None:
            self.nms_thresh = nms_thresh
        det_thresh = kwargs.get('det_thresh', None)
        if det_thresh is not None:
            self.det_thresh = det_thresh
        input_size = kwargs.get('input_size', None)
        if input_size is not None:
            if self.input_size is not None:
                print('warning: det_size is already set in detection model, ignore')
            else:
                self.input_size = input_size

    def detect(self, img, input_size = None, max_num=0, metric='default'):
        ori_im = img.copy()
        img = self.preprocess_img(img, self.image_shape)
        # print(img)
        if img is None:
            return None, 0
        starttime = time.time()
        y_bboxes_output, y_cls_output = self.session.run([self.box_output_name, self.cls_output_name], input_feed={self.input_name: img})
        # print(y_bboxes_output, y_cls_output)

        post_result = self.postprocess_op(ori_im, y_bboxes_output, y_cls_output)
        elapse = time.time() - starttime
        return post_result, None


