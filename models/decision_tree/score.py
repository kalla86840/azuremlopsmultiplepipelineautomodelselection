import joblib
import numpy as np

model = joblib.load('model.joblib')

def run(input_data):
    data = np.array(input_data['data'])
    return model.predict(data).tolist()