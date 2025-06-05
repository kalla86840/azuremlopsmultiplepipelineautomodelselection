import os
import pandas as pd
import json
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

models = {
    "LinearRegression": "predictions_linear.csv",
    "RandomForest": "predictions_rf.csv",
    "DecisionTree": "predictions_dt.csv"
}

ground_truth_path = "ground_truth.csv"
metrics_list = []

if not os.path.exists(ground_truth_path):
    print("❌ Missing ground truth file:", ground_truth_path)
    exit(1)

y_true = pd.read_csv(ground_truth_path).squeeze()

print("🔍 Comparing model metrics:")
print("-" * 40)

for model_name, pred_path in models.items():
    if not os.path.exists(pred_path):
        print(f"⚠️  Skipping {model_name}: Missing {pred_path}")
        continue

    y_pred = pd.read_csv(pred_path).squeeze()

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"✅ {model_name}")
    print(f"   R² Score     : {r2:.4f}")
    print(f"   MAE          : {mae:.4f}")
    print(f"   MSE          : {mse:.4f}")
    print("-" * 40)

    metrics_list.append({
        "model": model_name,
        "r2_score": r2,
        "mae": mae,
        "mse": mse
    })

# Save timestamped reports
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
metrics_dir = "reports"
os.makedirs(metrics_dir, exist_ok=True)

summary_csv = os.path.join(metrics_dir, f"metrics_report_{timestamp}.csv")
summary_json = os.path.join(metrics_dir, f"metrics_report_{timestamp}.json")

pd.DataFrame(metrics_list).to_csv(summary_csv, index=False)
with open(summary_json, "w") as f:
    json.dump(metrics_list, f, indent=2)

# Determine best model
if metrics_list:
    ranked = sorted(metrics_list, key=lambda x: (-x["r2_score"], x["mae"]))
    best_model = ranked[0]["model"]
    print(f"🏆 Best model based on R² and MAE: {best_model}")
    with open("best_model.txt", "w") as f:
        f.write(best_model)
else:
    print("❌ No valid models were evaluated.")
    exit(1)