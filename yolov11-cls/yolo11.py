# import
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-cls.pt")

results = model.train(data="food", epochs=20, imgsz=64)
