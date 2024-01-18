# -*- coding: utf-8 -*-
from ultralytics import YOLO

# 预测过程：
model = YOLO("runs/detect/train228/weights/best.pt")  # 权重
model.predict(source='/home/grr/xwj/datasets/VisDrone/VisDrone2019-DET-val/images',
              name="predict_61(3)-",imgsz=800,
              save=True, save_txt=True, hide_conf=True,device=0)

#或者也可以如下操作
# for i in model.predict(source='/home/user/chenhao/datasets/VisDrone2019-DET/VisDrone2019-DET-test-dev/images', conf=0.6,
#               name="predict_test",stream=True,
#               save=True, save_txt=True, hide_conf=True):
#     print(i)  #这里可以对i进行一些操作
