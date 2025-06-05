import os
import joblib
from azureml.core import Workspace, Model

def register_model(model_name, model_path, workspace_config="config.json"):
    ws = Workspace.from_config(path=workspace_config)
    Model.register(workspace=ws,
                   model_path=model_path,
                   model_name=model_name)
    print(f"Model '{model_name}' registered successfully.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python register_model.py <model_name> <model_path>")
        exit(1)
    register_model(sys.argv[1], sys.argv[2])