import tensorflow as tf # 2.4.1 
from tensorflow.keras.models import load_model

model = load_model('') # .model or .h5 file

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("activity-lite.tflite", "wb") as f:
    f.write(tflite_model)
