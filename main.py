# Importing all the necessary libraries
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
import uvicorn
import numpy as np
import tensorflow as tf
from keras.applications.vgg16 import preprocess_input
import os
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

# creating an object for fast api 
app = FastAPI()

# to render HTML templates in FastAPI we use Jinja2
# creating object for this and specifying the directory to look for HTML files
templates = Jinja2Templates(directory="templates")

# Loading the saved model 
MODEL = tf.keras.models.load_model("model_vgg16.h5")

# creating and array to decode the prediction given by our model
CLASS_NAMES = ["Normal", "Pneumonia"]

# This function takes the file image path and model path 
# does all the preprocessing, converts to array and
# returns the predictions 
def model_predict(img_path, MODEL):
    img = tf.keras.utils.load_img(img_path, target_size = (224,224))
    img = tf.keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img_data = preprocess_input(img)
    preds = MODEL.predict(img_data)
    return preds

# To save the uploaded image in a temp directory, and then return the path of the uploaded image 
# the uploaded image will be stored in a temprory library with a random name
# this will return the path of that image
def save_upload_file_tmp(upload_file: UploadFile) -> Path:
    try:
        suffix = Path(upload_file.filename).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
            tmp_path = Path(tmp.name)
    finally:
        upload_file.file.close()
    return tmp_path

# The route to check if the API is working 
@app.get("/ping")
def ping():
    return "pinging!"

# The Home Route return the route when the website loads up
@app.get("/")
# def home(request: Request):
#     return templates.TemplateResponse("base.html", {'request':request})
def home():
    return{"Please route to /predict for prediction"}

# Route for taking input as image 
# doing preprocessing and feeding image to the model 
# and getting the prediction
@app.post("/predict")
async def predict(
    # taking input of file in our case image
    file: UploadFile = File(...)
):
    # calling the function and getting the temp path of the file
    filepath = save_upload_file_tmp(file)
    print(filepath)
    # feeding the model predict function file path and loded model to get the prediction
    predictions = model_predict(filepath, MODEL)
    # after we get the prediction we want to remove the uploaded image to conserver space
    filepath.unlink()

    # finding the max of the nd array retruned by the model prediction and
    # using class names to find which class it belongs to
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    # calculating the accuracy/confidence of our prediction
    confidence = np.max(predictions[0])

    # retruning the predicted classes, Pneumonia or normal
    # returning the confidence
    return{
        'class' : predicted_class,
        'confidence' : float(confidence)*100
    }

# To run the app using uvicorn on local host giving address and port number
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
