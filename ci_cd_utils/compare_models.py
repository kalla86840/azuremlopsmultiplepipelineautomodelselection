import json
import os
import sys

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

    # Select the best model by highest R², then lowest RMSE
    best_model = max(metrics.items(), key=lambda x: (x[1].get("r2", 0), -x[1].get("rmse", float("inf"))))[0]

    report = {
        "models": metrics,
        "best_model": best_model
    }

    os.makedirs("reports", exist_ok=True)
    with open(f"reports/metrics_report_{dataset}.json", "w") as f:
        json.dump(report, f, indent=2)

    with open(f"outputs/{dataset}_best_model.txt", "w") as f:
        f.write(best_model)

    print(f"Best model for {dataset}: {best_model}")

if __name__ == "__main__":
    main()