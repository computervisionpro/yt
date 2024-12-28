# import
from ultralytics import YOLO

# Load a model
model = YOLO("best.pt")
foods = model.names
print(foods)

results = model("./food/val/pizza/3475871.jpg")


print("\n\nThe food in image is:")
class_id = results[0].probs.data.argmax()
print(foods[class_id.item()])