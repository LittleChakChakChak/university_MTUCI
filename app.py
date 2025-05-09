from fastapi import FastAPI, File, UploadFile
import pandas as pd
import joblib
from io import BytesIO
import os

!pip install python-multipart

app = FastAPI()

# Укажите путь к модели (обязательно сохраните её туда заранее)
model_path = '/content/drive/MyDrive/MTUCI/laptop_price_model.pkl'
model = joblib.load(model_path)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(BytesIO(content))
    predictions = model.predict(df)
    return {"predictions": predictions.tolist()}
