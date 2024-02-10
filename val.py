# -*- coding: utf-8 -*-
from ultralytics import YOLO

model = YOLO("runs/detect/train80/weights/best.pt")  # 权重地址
model.val(name='word68(1)',imgsz=800,batch=1,device=0,data="ultralytics/datasets/VisDrone_205.yaml")  # 数和训练用到的一样
