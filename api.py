import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_URI = "models:/m-61f5801e9fe74fcd978af95302355f53"

mlflow.set_tracking_uri("http://127.0.0.1:5000")
model = mlflow.sklearn.load_model(MODEL_URI)

app = FastAPI(title="Lidar Regression API")


class PredictionInput(BaseModel):
    range: float = Field(allow_inf_nan=False)


@app.get("/health")
def health():
    return {"status": "ready", "model_uri": MODEL_URI}


@app.post("/predict")
def predict(inputs: PredictionInput):
    data = pd.DataFrame({"range": [inputs.range]})
    prediction = float(model.predict(data)[0])

    return {
        "range": inputs.range,
        "predicted_logratio": prediction,
    }
