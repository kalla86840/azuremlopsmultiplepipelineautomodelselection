import sys
import pandas as pd
import json
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def train_and_save(dataset_path):
    df = pd.read_csv(dataset_path)
    dataset = os.path.splitext(os.path.basename(dataset_path))[0]

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    os.makedirs("outputs", exist_ok=True)
    with open(f"outputs/{dataset}_RandomForest_y_true.json", "w") as f:
        json.dump(y_test.tolist(), f)
    with open(f"outputs/{dataset}_RandomForest_y_pred.json", "w") as f:
        json.dump(y_pred.tolist(), f)

if __name__ == "__main__":
    train_and_save(sys.argv[1])