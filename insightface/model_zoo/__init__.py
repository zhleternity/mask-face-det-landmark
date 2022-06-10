# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# @File: face_analysis.py
# @Date Modified: 2022/06/09 12:30:40
# @Author: zhouhl-b

from .model_zoo import get_model
from .arcface_onnx import ArcFaceONNX
from .retinaface import RetinaFace
from .scrfd import SCRFD
from .landmark import Landmark
from .attribute import Attribute
from .mask.maskface import MaskFace
