import os
from azureml.core import Workspace, Model, Environment
from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.model import InferenceConfig

# Load workspace from config.json
import json
with open("config/cars_config.json") as f:
    config = json.load(f)

ws = Workspace(subscription_id=config["subscription_id"],
               resource_group=config["resource_group"],
               workspace_name=config["workspace_name"])

# Read best model name
with open("best_model.txt", "r") as f:
    best_model_name = f.read().strip().lower()

model_name_map = {
    "linearregression": "linear_regression_model",
    "randomforest": "random_forest_model",
    "decisiontree": "decision_tree_model"
}

model_name = model_name_map.get(best_model_name.replace(" ", "").lower())
if not model_name:
    raise ValueError(f"Unsupported model name in best_model.txt: {best_model_name}")

try:
    model = Model(ws, name=model_name)
except Exception as e:
    raise Exception(f"Model '{model_name}' not found in workspace: {str(e)}")

# Prepare inference config
env = Environment(name="deploy-env")
inference_config = InferenceConfig(entry_script=f"models/{best_model_name.lower().replace(' ', '_')}/score.py",
                                   environment=env)

# Define deployment config
deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

# Deploy
service_name = f"{best_model_name.lower().replace(' ', '-')}-service"
service = Model.deploy(workspace=ws,
                       name=service_name,
                       models=[model],
                       inference_config=inference_config,
                       deployment_config=deployment_config,
                       overwrite=True)
service.wait_for_deployment(show_output=True)
print(f"✅ Deployed service at: {service.scoring_uri}")