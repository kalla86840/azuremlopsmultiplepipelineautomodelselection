import json
import os
import sys
from datetime import datetime
import shutil

def load_metrics(model_name, dataset):
    path = f"outputs/{dataset}_{model_name}_metrics.json"
    with open(path, "r") as f:
        return json.load(f)

def main():
    if len(sys.argv) != 2:
        print("Usage: python compare_models.py <dataset>")
        sys.exit(1)

    dataset = sys.argv[1]
    model_names = ["LinearRegression", "DecisionTree", "RandomForest"]
    metrics = {}

    for model in model_names:
        try:
            metrics[model] = load_metrics(model, dataset)
        except Exception as e:
            print(f"Error loading metrics for {model}: {e}")
            continue

    if not metrics:
        print("No metrics found. Exiting.")
        sys.exit(1)

    best_model = max(metrics.items(), key=lambda x: (x[1].get("r2", 0), -x[1].get("rmse", float("inf"))))[0]
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    report = {
        "models": metrics,
        "best_model": best_model,
        "timestamp": timestamp
    }

    os.makedirs("reports", exist_ok=True)
    full_report_path = f"reports/metrics_report_{dataset}_{timestamp}.json"
    with open(full_report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Optionally keep the latest version always in place
    latest_path = f"reports/metrics_report_{dataset}.json"
    shutil.copy(full_report_path, latest_path)

    with open(f"outputs/{dataset}_best_model.txt", "w") as f:
        f.write(best_model)

    print(f"Best model for {dataset}: {best_model}")

if __name__ == "__main__":
    main()