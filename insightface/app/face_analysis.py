# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# @File: face_analysis.py
# @Date Modified: 2022/06/09 12:30:40
# @Author: zhouhl-b


from __future__ import division

import glob
import os.path as osp

import numpy as np
import onnxruntime
from numpy.linalg import norm

from ..model_zoo import model_zoo
from ..utils import DEFAULT_MP_NAME, ensure_available
from .common import Face

__all__ = ['FaceAnalysis']

class FaceAnalysis:
    def __init__(self, name=DEFAULT_MP_NAME, root='/mnt/data/workspace/work_algos/Algos/glodon/bgy-fangfeidan/code/face-mask-det-align-220607/mask-det-landmark-pipeline', allowed_modules=None, **kwargs): # '~/.insightface'
        onnxruntime.set_default_logger_severity(3)
        self.models = {}
        self.model_dir = ensure_available('models', name, root=root)
        print(self.model_dir)
        onnx_files = glob.glob(osp.join(self.model_dir, '*.onnx'))
        # print(onnx_files)
        onnx_files = sorted(onnx_files)
        for onnx_file in onnx_files:
            model = model_zoo.get_model(onnx_file, **kwargs)
            # print(model.taskname)
            if model is None:
                print('model not recognized:', onnx_file)
            elif allowed_modules is not None and model.taskname not in allowed_modules:
                print('model ignore:', onnx_file, model.taskname)
                del model
            elif model.taskname not in self.models and (allowed_modules is None or model.taskname in allowed_modules):
                print('find model:', onnx_file, model.taskname, model.input_shape, model.input_mean, model.input_std)
                self.models[model.taskname] = model
            else:
                print('duplicated model task type, ignore:', onnx_file, model.taskname)
                del model
        assert 'detection' in self.models
        # print(self.models)
        self.det_model = self.models['detection']
        self.id2class = {0: 'Mask', 1: 'NoMask'}


    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640)):
        self.det_thresh = det_thresh
        assert det_size is not None
        print('set det-size:', det_size)
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname=='detection':
                model.prepare(ctx_id, input_size=det_size, det_thresh=det_thresh)
            else:
                model.prepare(ctx_id)


    def get(self, img, max_num=0):
        bboxes, kpss = self.det_model.detect(img,
                                             max_num=max_num,
                                             metric='default')

        # print(bboxes, kpss)
        if bboxes.shape[0] == 0:
            return []
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            for taskname, model in self.models.items():
                if taskname=='detection':
                    continue
                model.get(img, face)
            ret.append(face)
            # print(ret)
        return ret

    def run(self, img, max_num=0):
        bboxes, kpss = self.det_model.detect(img,
                                             max_num=max_num,
                                             metric='default')

        # print(bboxes, kpss)
        if len(bboxes) == 0:
            return []
        ret = []
        for i in range(len(bboxes)):
            bbox = bboxes[i][2:6]
            det_score = bboxes[i][1]
            det_label = self.id2class[bboxes[i][0]]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score, det_label=det_label)
            # print(face)
            for taskname, model in self.models.items():
                if taskname=='detection':
                    continue
                model.get(img, face)
            ret.append(face)
            print(ret)
        return ret

    def draw_on(self, img, faces):
        import cv2
        dimg = img.copy()
        for i in range(len(faces)):
            face = faces[i]
            box = face.bbox.astype(np.int)
            color = (0, 0, 255)
            cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
            if face.kps is not None:
                kps = face.kps.astype(np.int)
                #print(landmark.shape)
                for l in range(kps.shape[0]):
                    color = (0, 0, 255)
                    if l == 0 or l == 3:
                        color = (0, 255, 0)
                    cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color,
                               2)
            if face.gender is not None and face.age is not None:
                cv2.putText(dimg,'%s,%d'%(face.sex,face.age), (box[0]-1, box[1]-4),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),1)

            #for key, value in face.items():
            #    if key.startswith('landmark_3d'):
            #        print(key, value.shape)
            #        print(value[0:10,:])
            #        lmk = np.round(value).astype(np.int)
            #        for l in range(lmk.shape[0]):
            #            color = (255, 0, 0)
            #            cv2.circle(dimg, (lmk[l][0], lmk[l][1]), 1, color,
            #                       2)
        return dimg

