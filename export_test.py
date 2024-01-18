# -*- coding: utf-8 -*-
from ultralytics import YOLO

#将训练好的网络模型导出为onnx格式
model = YOLO("path/best.pt")

model.export(format="onnx")