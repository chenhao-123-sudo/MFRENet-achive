# -*- coding: utf-8 -*-
from ultralytics import YOLO

# 如果要验证自己的训练结果，如下：ultralytics import YOLO
# 406验证
# model = YOLO("runs/detect/train256/weights/best.pt")  # 权重地址
# model.val(name='word66(1)',imgsz=800,batch=1,device=0)  # 参数和训练用到的一样


# 205-2验证
model = YOLO("runs/detect/train80/weights/best.pt")  # 权重地址
model.val(name='word68(1)',imgsz=800,batch=1,device=0,data="ultralytics/datasets/VisDrone_205.yaml")  # 数和训练用到的一样

# 304验证
# model = YOLO("yolov8s.pt")
# model = YOLO("runs/detect/yolov8_306/yolov8_306_22/weights/best.pt")  # 权重地址
# model.val(data="D:/Public/chenhao/ultralytics/ultralytics/datasets/VisDrone_304.yaml")   #参数和训练用到的一样


# print("map50-95:",model.box.map)
# print("map50:",model.box.map50)
# print("map75:",model.box.map75)
# print("map50-95 of each category:",model.box.maps)