import json
import numpy as np
import joblib
from sklearn.base import BaseEstimator

def init():
    global model
    model_path = "model.joblib"
    model = joblib.load(model_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)["data"]
        predictions = model.predict(data)
        return predictions.tolist()
    except Exception as e:
        return {"error": str(e)}