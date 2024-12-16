from ultralytics import YOLO
 
# Load the model.
model = YOLO('yolov10n.pt')
 
# Training.
results = model.train(
   data='data3.yaml',
   imgsz=448,
   epochs=40,
   batch=8,
   name='yolov10n_park_40e'
)
