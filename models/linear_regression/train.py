import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import sys

data_path = sys.argv[1] if len(sys.argv) > 1 else 'data/cars.csv'
df = pd.read_csv(data_path)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, 'model.joblib')