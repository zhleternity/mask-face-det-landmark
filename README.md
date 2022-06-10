# mask-face-det-landmark

基于InsightFace和ONNX实现的口罩人脸检测，属性输出，以及人脸landmark，主要涉及到两个模型：检测模型（2分支，输出是否戴口罩和人脸位置框），landmark模型（输出106个关键点）；

## 目录结构

```
mask-det-landmark-pipeline
├─ image_infer.py                           #主要inference代码
├─ insightface
│  ├─ __init__.py
│  ├─ app
│  │  ├─ __init__.py
│  │  ├─ common.py
│  │  ├─ face_analysis.py                   #
│  │  └─ mask_renderer.py
│  ├─ data
│  │  ├─ __init__.py
│  │  ├─ image.py
│  │  ├─ test0602                           # 测试图像
│  │  └─ test0607                           # 测试图像
│  ├─ model_zoo                             # 各个模块的模型类
│  │  ├─ __init__.py
│  │  ├─ arcface_onnx.py
│  │  ├─ attribute.py
│  │  ├─ landmark.py
│  │  ├─ mask                               #这里是我封装的口罩人脸的类
│  │  │  ├─ __init__.py
│  │  │  ├─ maskface.py
│  │  │  ├─ postprocess.py                  #后处理
│  │  │  └─ utils
│  │  │     ├─ __init__.py
│  │  │     ├─ anchor_decode.py
│  │  │     ├─ anchor_generator.py
│  │  │     └─ nms.py
│  │  ├─ model_store.py
│  │  ├─ model_zoo.py
│  │  ├─ retinaface.py
│  │  └─ scrfd.py
│  ├─ thirdparty
│  └─ utils
├─ models                                    #模型文件，.onnx
│  └─ buffalo_l
│     ├─ 2d106det.onnx
│     └─ maskface.onnx
└─ output
   ├─ t2.jpg

```

## 代码运行

### 环境准备

Python3.x
onnx
onnx-runtime等

其他库，可以在代码运行时，缺少什么即pip install 什么，即可；

### 运行

 python image_infer.py

## 替换模型

由于本代码基于onnx模型做推理，因此，模型要统一转换成onnx模型；

如果需要替换其他全新的模型，可以按照代码中maskface类来进行封装，主要实现prepare，detect方法；

需要修改的位置：

第一处：

app = FaceAnalysis(allowed_modules=['detection', 'landmark_2d_106'])  #此处需要添加对应的teskname

第二处： 

if len(outputs)>=5:
    return RetinaFace(model_file=self.onnx_file, session=session)
elif len(outputs) == 2:                                                #此处需要添加你自己已封装好的任务类，进行模型加载
    # print('#########################MASK#########################')
    return MaskFace(model_file=self.onnx_file, session=session)              # model_zoo.py

第三处：

face_analysis.py中需要按照你模型输出格式，来对应修改输出后处理；
 
bboxes, kpss = self.det_model.detect(img,
                                             max_num=max_num,
                                             metric='default')

# print(bboxes, kpss)