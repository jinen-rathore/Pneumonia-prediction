
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import os
import tensorflow as tf
import numpy as np

model = load_model('model_vgg16.h5')
img = tf.keras.utils.load_img('chest_xray/val/NORMAL/NORMAL2-IM-1442-0001.jpeg', target_size=(224, 224))
x = tf.keras.utils.img_to_array(img)
x = np.expand_dims(x, axis=0)
img_data = preprocess_input(x)
predictions = model.predict(img_data)

CLASS_NAMES = ["Normal", "Pneumonia"]

predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
confidence = np.max(predictions[0])

print(predicted_class)
print("confidence: ",confidence*100, "%")
