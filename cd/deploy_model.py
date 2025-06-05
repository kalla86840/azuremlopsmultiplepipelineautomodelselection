import os
import sys

def deploy_model(dataset):
    best_model_path = f"outputs/{dataset}_best_model.txt"

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model file not found: {best_model_path}")

    with open(best_model_path, "r") as f:
        best_model = f.read().strip()

    print(f"Deploying best model: {best_model} for dataset: {dataset}")
    # Stub: Replace with actual Azure ML deployment logic for the model
    # Example: azureml-deployment-code here
    # Currently, this is just a placeholder print statement
    print(f"Model '{best_model}' deployment for '{dataset}' is completed.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python deploy_model.py <dataset>")
        sys.exit(1)
    deploy_model(sys.argv[1])