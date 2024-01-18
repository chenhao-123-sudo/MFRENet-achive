# -*- coding: utf-8 -*-
from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()
    # load a model
    # model = YOLO("ultralytics/models/v8/yolov8_swin.yaml").train(**{'cfg': 'ultralytics/yolo/cfg/default_bake_304.yaml'})
    # model = YOLO("ultralytics/models/v8/yolov8_shuattr.yaml").train(**{'cfg': 'ultralytics/yolo/cfg/default_bake_304.yaml'})
    # model = YOLO("ultralytics/models/v8/yolov8_simam.yaml").train(**{'cfg': 'ultralytics/yolo/cfg/default_bake_304.yaml'})
    # model = YOLO("ultralytics/models/v8/yolov8_EMA.yaml").train(**{'cfg': 'ultralytics/yolo/cfg/default_bake_304.yaml'})
    # model = YOLO("ultralytics/models/v8/yolov8_DCN.yaml").train(**{'cfg': 'ultralytics/yolo/cfg/default_bake_304.yaml'})
    # model = YOLO("ultralytics/models/v8/yolov8_TripletAttention.yaml").train(**{'cfg': 'ultralytics/yolo/cfg/default_bake_304.yaml'})
    # model = YOLO("ultralytics/models/v8/yolov8_mymodel1.yaml").train(**{'cfg': 'ultralytics/yolo/cfg/default_bake_304.yaml'})

    model = YOLO("ultralytics/models/v8/yolov8_CSAT.yaml").load("D:/Public/chenhao/ultralytics/yolov8s.pt")
    model.train(data="D:/Public/chenhao/ultralytics/ultralytics/datasets/VisDrone_304.yaml",
                epochs=200, batch=4, workers=0, device=0, lr0=0.001)
