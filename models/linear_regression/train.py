import sys
import pandas as pd
import json
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def train_and_save(dataset):
    df = pd.read_csv(f"data/{dataset}.csv")
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    os.makedirs("outputs", exist_ok=True)
    with open(f"outputs/{dataset}_LinearRegression_y_true.json", "w") as f:
        json.dump(y_test.tolist(), f)
    with open(f"outputs/{dataset}_LinearRegression_y_pred.json", "w") as f:
        json.dump(y_pred.tolist(), f)

if __name__ == "__main__":
    dataset = sys.argv[1]
    train_and_save(dataset)