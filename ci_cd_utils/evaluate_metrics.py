import json
import sys
import os
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

def main():
    if len(sys.argv) != 3:
        print("Usage: python evaluate_metrics.py <dataset> <model_name>")
        sys.exit(1)

    dataset = sys.argv[1]
    model_name = sys.argv[2]

    y_true_path = f"outputs/{dataset}_{model_name}_y_true.json"
    y_pred_path = f"outputs/{dataset}_{model_name}_y_pred.json"

    if not os.path.exists(y_true_path) or not os.path.exists(y_pred_path):
        print(f"Missing prediction or ground truth files for {model_name} on {dataset}")
        sys.exit(1)

    with open(y_true_path, "r") as f:
        y_true = json.load(f)
    with open(y_pred_path, "r") as f:
        y_pred = json.load(f)

    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)

    metrics = {
        "r2": round(r2, 4),
        "rmse": round(rmse, 4)
    }

    os.makedirs("outputs", exist_ok=True)
    with open(f"outputs/{dataset}_{model_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved metrics for {model_name} on {dataset}: {metrics}")

if __name__ == "__main__":
    main()