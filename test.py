from fastapi import FastAPI, File, UploadFile, Request
import uvicorn
import numpy as np
import tensorflow as tf
from keras.applications.vgg16 import preprocess_input
import os
from fastapi.templating import Jinja2Templates
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


templates = Jinja2Templates(directory="templates/")

MODEL = tf.keras.models.load_model("model_vgg16.h5")

CLASS_NAMES = ["Normal", "Pneumonia"]

def model_predict(img_path, MODEL):
    img = tf.keras.utils.load_img(img_path, target_size = (224,224))
    img = tf.keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img_data = preprocess_input(img)
    preds = MODEL.predict(img_data)
    return preds

def save_upload_file_tmp(upload_file: UploadFile) -> Path:
    try:
        suffix = Path(upload_file.filename).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
            tmp_path = Path(tmp.name)
    finally:
        upload_file.file.close()
    return tmp_path

@app.get("/ping")
def ping():
    return "pinging!"

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {'request':request})

# @app.post("/")
# def home(request: Request, file: UploadFile = File(...)):
#     return templates.TemplateResponse("base.html", {'request':request})


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # basepath = os.path.dirname(__file__)
    # file_path = os.path.join(basepath,"chest_xray/val", file.filename)
    filepath = save_upload_file_tmp(file)
    print(filepath)
    predictions = model_predict(filepath, MODEL)

    filepath.unlink()
    # predictions = model_predict(file_path, MODEL)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return{
        'class' : predicted_class,
        'confidence' : float(confidence)*100
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
