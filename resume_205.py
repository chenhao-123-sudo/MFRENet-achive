from ultralytics import YOLO

model = YOLO("runs/detect/train89/weights/last.pt")

model.train(data="ultralytics/datasets/VisDrone_205.yaml",
            resume=True)
