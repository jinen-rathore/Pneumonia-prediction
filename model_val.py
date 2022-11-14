
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import os
import tensorflow as tf
import numpy as np

model = load_model('./models/trained_model.h5')
images = ["normal1.jpeg", "normal2.jpeg", "pneumonia1.jpeg", "pneumonia2.jpeg"]
for image in images:
    img = tf.keras.utils.load_img('./Images/'+image, target_size=(64,64,3))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    img_data = preprocess_input(x)
    predictions = model.predict(img_data)

    CLASS_NAMES = ["Normal", "Pneumonia"]

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    print(predicted_class)
    print("confidence: ",confidence*100, "%")
