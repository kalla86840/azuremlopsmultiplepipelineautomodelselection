from azureml.core import Workspace, Model

ws = Workspace.from_config(path='config/cars_config.json')
model = Model.register(model_path='model.joblib',
                       model_name='linear_regression_model',
                       workspace=ws)
print(f"Model registered: {model.name}, ID: {model.id}")