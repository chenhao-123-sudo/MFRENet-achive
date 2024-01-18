from ultralytics import YOLO

model = YOLO("ultralytics/models/v8/yolov8_CSAT.yaml").load("./yolov8m.pt")
model = YOLO("/home/wenzhuyang/ch/ultralytics/runs/detect/train22/weights/last.pt")

model.train(data="/home/wenzhuyang/ch/ultralytics/ultralytics/datasets/VisDrone_cs.yaml",
            epochs=200, batch=4, workers=16, device=0, lr0=0.001,resume=True)