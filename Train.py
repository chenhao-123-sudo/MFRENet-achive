from ultralytics import YOLO

model = YOLO("ultralytics/models/v8/MFRENet.yaml").load("./yolov8m.pt")
model.train(data="ultralytics/datasets/VisDrone_406.yaml",
            epochs=150, batch=2, imgsz=800, lr0=0.001,patience=50)

