# -*- coding: utf-8 -*-
from ultralytics import YOLO

model = YOLO("ultralytics/models/v8/yolov8.yaml").load("./yolov8n.pt")
model.train(data="/home/user/chenhao/ultralytics/ultralytics/datasets/VisDrone_406.yaml",
            epochs=500, batch=2, imgsz=800, lr0=0.001,patience=50)

