from ultralytics import YOLO
 
# Load the model.
model = YOLO('yolov8n.pt')
 
# Training.
results = model.train(
   data='data.yaml',
   imgsz=416,
   epochs=40,
   batch=8,
   name='yolov8n_park_20e'
)
