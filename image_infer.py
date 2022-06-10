# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# @File: face_analysis.py
# @Date Modified: 2022/06/09 14:30:40
# @Author: zhouhl-b

import cv2
import numpy as np
import os
import time
import os.path as osp
# import insightface
from insightface.app import FaceAnalysis
from insightface.data import getFilename

if __name__ == '__main__':
    app = FaceAnalysis(allowed_modules=['detection', 'landmark_2d_106'])
    app.prepare(ctx_id=0, det_size=(360, 360))
    save_dir = './output/'
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    input_path = './insightface/data/test0602'
    filenames = getFilename(input_path)
    total_num = len(filenames)
    total_time = 0.0
    for file in filenames:
        img = cv2.imread(file)
        tim = img.copy()
        img_n = osp.basename(file)
        tt = time.time()
        faces = app.run(img)
        print('Using Time: ', (time.time() - tt)*1000)
        total_time += (time.time() - tt)*1000
        
        color = (0, 0, 255)#(200, 160, 75)
        for face in faces:
            # print(face)
            lmk = face.landmark_2d_106
            lmk = np.round(lmk).astype(np.int)
            for i in range(lmk.shape[0]):
                p = tuple(lmk[i])
                cv2.rectangle(tim, (face.bbox[0], face.bbox[1]), (face.bbox[2], face.bbox[3]), (0, 255, 0), 2)
                cv2.circle(tim, p, 1, color, 1, cv2.LINE_AA)
                cv2.putText(tim, "%s: %.2f" % (face.det_label, face.det_score), (face.bbox[0] + 2, face.bbox[1] - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
        cv2.imwrite(save_dir + img_n, tim)

    print('Everage Time: ', total_time / total_num)


    
