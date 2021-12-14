import cv2
import pickle
import numpy as np
import tensorflow as tf

mean = np.array([123.68, 116.779, 103.939], dtype="float32") # image net rgb mean
#----
###### TO DOWNLOAD THE TFLITE MODEL I USED (95 MB): https://drive.google.com/file/d/1ycHzSzIm1ZO-CYfNyQtukLmOQVOfBVK3/view?usp=sharing
#----
interpreter = tf.lite.Interpreter(model_path="activity-lite.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)

labels = pickle.loads(open('activity-labels.pickle', 'rb').read())

image = cv2.imread("p-024.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224,224)).astype("float32")
image -= mean

interpreter.set_tensor(input_details[0]['index'], np.expand_dims(image, axis=0))
interpreter.invoke()

pred = interpreter.get_tensor(output_details[0]['index'])
i = np.argmax(pred)
label = labels.classes_[i]

print(label)
