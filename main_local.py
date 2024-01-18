# -*- coding: utf-8 -*-
from ultralytics import YOLO
import multiprocessing


if __name__ == '__main__':
    multiprocessing.freeze_support()
    # 如果要中断续训，这个文件里的内容不需要修改，只需要修改trainer.py里面的内容，修改def check_resume(self):和def resume_training(self, ckpt):两个方法里面的内容。
    # 其中def check_resume(self):里添将resume = self.args.resume注释，添加resume = 'runs/detect/train/weights/last.pt'需要续训得last.pt文件
    # def resume_training(self, ckpt):里面直接添加ckpt = torch.load('runs/detect/train/weights/last.pt')需要续训得last.pt文件
    # 需要正常训练之前记得改回原样

    # 另一种训练方式
    # 这里如果需要预权重就写你的权重文件地址，没有预权重写cfg地址，写一个就够了
    # model = YOLO("ultralytics/models/v8/yolov8_mymodel1.yaml").load("/home/user/chenhao/ultralytics/yolov8s.pt")
    # model.train(data="/home/user/chenhao/ultralytics/ultralytics/datasets/VisDrone_406.yaml",
    #             epochs=100, imgsz=640,batch=4,worker=0,device=0)

    # load a model
    # model = YOLO("ultralytics/models/v8/yolov8_swin.yaml").train(**{'cfg': 'ultralytics/yolo/cfg/default_bake_406.yaml'})
    # model = YOLO("ultralytics/models/v8/yolov8_shuattr.yaml").train(**{'cfg': 'ultralytics/yolo/cfg/default_bake_406.yaml'})
    # model = YOLO("ultralytics/models/v8/yolov8_simam.yaml").train(**{'cfg': 'ultralytics/yolo/cfg/default_bake_406.yaml'})
    # model = YOLO("ultralytics/models/v8/yolov8_EMA.yaml").train(**{'cfg': 'ultralytics/yolo/cfg/default_bake_406.yaml'})  #暂时不能运行
    # model = YOLO("ultralytics/models/v8/yolov8_DCN.yaml").train(**{'cfg': 'ultralytics/yolo/cfg/default_bake_406.yaml'})
    # model = YOLO("ultralytics/models/v8/yolov8.yaml").train(**{'cfg': 'ultralytics/yolo/cfg/default_bake_406.yaml'})
    # model = YOLO("ultralytics/models/v8/yolov8_Biformer.yaml").train(**{'cfg': 'ultralytics/yolo/cfg/default_bake_406.yaml'})


    # model = YOLO("ultralytics/models/v8/yolov8_mymodel.yaml").train(**{'cfg': 'ultralytics/yolo/cfg/default_bake_406.yaml'})
    # model = YOLO("ultralytics/models/v8/yolov8_mymodel1.yaml").train(**{'cfg': 'ultralytics/yolo/cfg/default_bake_406.yaml'})

    # model = YOLO("ultralytics/models/v8/yolov8.yaml").train(**{'cfg': 'ultralytics/yolo/cfg/default_bake_406.yaml'})
    # model = YOLO("ultralytics/models/v8/yolov8_TripletAttention.yaml").train(**{'cfg': 'ultralytics/yolo/cfg/default_bake_406.yaml'})

    # model = YOLO("ultralytics/models/v8/yolov8_mymodel1.yaml").load("/home/user/chenhao/ultralytics/yolov8s.pt")
    model = YOLO("ultralytics/models/v8/yolov8_CSAT.yaml").load("./yolov8m.pt")

    model.train(data="ultralytics/datasets/VisDrone_local.yaml",
                epochs=200, batch=2, workers=0, lr0=0.001,device='cpu',imgsz=640)
