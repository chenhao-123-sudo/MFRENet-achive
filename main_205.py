# -*- coding: utf-8 -*-
from ultralytics import YOLO
import multiprocessing

# if __name__ == '__main__':
#     multiprocessing.freeze_support()
    # load a model
    # model = YOLO("ultralytics/models/v8/yolov8_swin.yaml").train(**{'cfg': 'ultralytics/yolo/cfg/default_bake_406.yaml'})
    # model = YOLO("ultralytics/models/v8/yolov8_swin.yaml").train(**{'cfg': 'ultralytics/yolo/cfg/default_bake_304.yaml'})
    # model = YOLO("ultralytics/models/v8/yolov8_shuattr.yaml").train(**{'cfg': 'ultralytics/yolo/cfg/default_bake_406.yaml'})
    # model = YOLO("ultralytics/models/v8/yolov8_shuattr.yaml").train(**{'cfg': 'ultralytics/yolo/cfg/default_bake_304.yaml'})
    # model = YOLO("ultralytics/models/v8/yolov8_simam.yaml").train(**{'cfg': 'ultralytics/yolo/cfg/default_bake_406.yaml'})
    # model = YOLO("ultralytics/models/v8/yolov8_simam.yaml").train(**{'cfg': 'ultralytics/yolo/cfg/default_bake_304.yaml'})
    # model = YOLO("ultralytics/models/v8/yolov8_EMA.yaml").train(**{'cfg': 'ultralytics/yolo/cfg/default_bake_304.yaml'})
    # model = YOLO("ultralytics/models/v8/yolov8_EMA.yaml").train(**{'cfg': 'ultralytics/yolo/cfg/default_bake_406.yaml'})  #暂时不能运行
    # model = YOLO("ultralytics/models/v8/yolov8_DCN.yaml").train(**{'cfg': 'ultralytics/yolo/cfg/default_bake_304.yaml'})
    # model = YOLO("ultralytics/models/v8/yolov8_DCN.yaml").train(**{'cfg': 'ultralytics/yolo/cfg/default_bake_406.yaml'})
    # model = YOLO("ultralytics/models/v8/yolov8_mymodel.yaml").train(**{'cfg': 'ultralytics/yolo/cfg/default_bake_304.yaml'})
    # model = YOLO("ultralytics/models/v8/yolov8_mymodel1.yaml").train(**{'cfg': 'ultralytics/yolo/cfg/default_bake_205.yaml'})
    # model = YOLO("ultralytics/models/v8/yolov8_mymodel.yaml").train(**{'cfg': 'ultralytics/yolo/cfg/default_bake_406.yaml'})
    # model = YOLO("ultralytics/models/v8/yolov8.yaml").train(**{'cfg': 'ultralytics/yolo/cfg/default_bake_406.yaml'})


model = YOLO("ultralytics/models/v8/yolov8_CSAT.yaml").load("./yolov8m.pt")
model.train(data="ultralytics/datasets/VisDrone_205.yaml",
            epochs=150, batch=2, imgsz=800, lr0=0.001,patience=50)


# model = YOLO("ultralytics/models/v8/yolov8.yaml").load("./yolov8m.pt")
# model.train(data="ultralytics/datasets/VisDrone_205.yaml",
#             batch=6, imgsz=800)

# model = YOLO("ultralytics/models/v8/yolov8_CSAT.yaml").load("/home/xwj/chenhao/ultralytics/yolov8m.pt")
# model.train(data="/home/xwj/chenhao/ultralytics/ultralytics/datasets/coco.yaml",
#             epochs=300, batch=8, workers=0, device=0, lr0=0.001)