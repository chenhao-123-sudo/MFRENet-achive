from ultralytics import YOLO


model = YOLO("runs/detect/train274/weights/last.pt")

model.train(data="ultralytics/datasets/VisDrone_406.yaml",
            resume=True)
