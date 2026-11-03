import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Iris Classifier API", version="1.0.0")

model = joblib.load("model.pkl")
CLASS_NAMES = ["setosa", "versicolor", "virginica"]


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, examples=[5.1])
    sepal_width: float = Field(..., gt=0, examples=[3.5])
    petal_length: float = Field(..., gt=0, examples=[1.4])
    petal_width: float = Field(..., gt=0, examples=[0.2])


@app.get("/")
def root():
    return {"message": "Iris Classifier API is running. Open /docs to test it."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(features: IrisFeatures):
    x = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]])
    pred = int(model.predict(x)[0])
    proba = model.predict_proba(x)[0]
    return {
        "predicted_class": pred,
        "predicted_species": CLASS_NAMES[pred],
        "confidence": round(float(proba[pred]), 4),
    }