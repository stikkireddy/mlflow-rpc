import os
import shutil
from pathlib import Path

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from mlflow.entities import Experiment
from mlflow.entities.model_registry import ModelVersion
from pathspec import PathSpec
from pathspec.patterns import GitWildMatchPattern

from mlrpc.flavor import FastAPIFlavor


def get_home_directory(ws_client: WorkspaceClient) -> Path:
    user = ws_client.current_user.me().user_name
    return Path("/Users") / str(user)


def get_or_create_mlflow_experiment(ws_client: WorkspaceClient, experiment_name: str) -> Experiment:
    experiment_path = get_home_directory(ws_client) / experiment_name
    experiment_path_str = str(experiment_path)
    exp = mlflow.get_experiment_by_name(experiment_path_str)
    if exp is None:
        print("Creating experiment")
        exp = mlflow.create_experiment(experiment_path_str)
        return mlflow.get_experiment(exp)
    return exp


def ensure_3_parts(name: str) -> bool:
    return len(name.split(".")) == 3


def save_model(experiment: Experiment, app: FastAPIFlavor, uc_model_path: str, run_name: str = "deployment") -> ModelVersion:
    if not ensure_3_parts(uc_model_path):
        raise ValueError \
            (f"Model path must be in the format 'catalog_name.schema_name.model_name' but got {uc_model_path}")

    src = Path(app.local_app_dir)
    dest = Path("/Users/sri.tikkireddy/PycharmProjects/mlflow-rpc/tmp")
    ignore_file = src / ".gitignore"
    ignore_file = ignore_file if ignore_file.exists() else None
    copy_files(app.local_app_dir, dest, ignore_file)
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name) as run:
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=app,
            signature=app.signature(),
            artifacts={app.code_key: str(dest)},
        )
    return mlflow.register_model(f"runs:/{run.info.run_id}/model", uc_model_path)


def copy_files(src, dest, ignore_file=None):
    spec = None
    src_dir = str(src)
    dest_dir = str(dest)
    if ignore_file is not None:
        ignore_file = str(ignore_file)
        with open(ignore_file, 'r') as f:
            gitignore = f.read()
        spec = PathSpec.from_lines(GitWildMatchPattern, gitignore.splitlines())

    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if ignore_file is None:
                shutil.copy(str(os.path.join(root, file)), dest_dir)
                continue

            if not spec.match_file(str(os.path.join(root, file))):
                shutil.copy(str(os.path.join(root, file)), dest_dir)


def deploy_serving_endpoint(ws_client: WorkspaceClient, endpoint_name: str, uc_model_path: str, model_version: ModelVersion):
    return ws_client.serving_endpoints.create_and_wait(
        name=endpoint_name,
        config=EndpointCoreConfigInput(
            name=endpoint_name,
            served_entities=[
                ServedEntityInput(
                    entity_version=model_version.version,
                    entity_name=uc_model_path,
                    workload_size="Small",
                    scale_to_zero_enabled=True,
                )
            ],
        )
    )
