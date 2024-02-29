import os
import shutil
from pathlib import Path
from typing import Optional, List

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


def default_mlrpc_libs():
    return ["mlflow==2.9.2",
            "cloudpickle==2.0.0",
            "mlflow-skinny==2.9.2",
            "fastapi==0.110.0",
            "pandas==1.5.3",
            "databricks-sdk==0.20.0",
            "httpx==0.27.0",
            "pathspec==0.12.1",
            "click",
            "python-dotenv",
            "mlrpc"]


def ensure_mlflow_installation(pip_reqs: List[str]):
    if any(["mlflow==" not in pip_req for pip_req in pip_reqs]):
        mlflow_pip_install = None
        for pip_req in pip_reqs:
            if pip_req.startswith("mlflow-skinny=="):
                mlflow_pip_install = pip_req.replace("mlflow-skinny", "mlflow")
                break
        if mlflow_pip_install is None:
            raise ValueError("mlflow-skinny==<version> must be present in pip_requirements")
        pip_reqs.append(mlflow_pip_install)
    return pip_reqs

def build_proper_requirements(file_path: Path) -> List[str]:
    req_libs = ensure_mlflow_installation(default_mlrpc_libs())
    if file_path.exists() is False:
        return req_libs

    print("Found requirements.txt using this")
    from pkg_resources import parse_requirements
    prebuilt_reqs = [lib.split("==")[0] for lib in req_libs]

    with open(file_path, 'r') as file:
        lines = file.readlines()
        # Ignore comments and empty lines
        lines = [line.strip() for line in lines if not line.startswith('#') and line.strip() != '']
        requirements = [str(req) for req in parse_requirements(lines) if req.name not in prebuilt_reqs]
        return req_libs + requirements

def save_model(
        ws_client: WorkspaceClient,
        experiment: Experiment,
        app: FastAPIFlavor,
        uc_model_path: str,
        run_name: str = "deployment",
        dest_path: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        latest_alias_name: Optional[str] = "current",
) -> ModelVersion:
    if not ensure_3_parts(uc_model_path):
        raise ValueError \
            (f"Model path must be in the format 'catalog_name.schema_name.model_name' but got {uc_model_path}")

    aliases = aliases or []
    aliases.append(latest_alias_name)
    aliases = list(set(aliases))
    src = Path(app.local_app_dir)
    dest = Path(dest_path)
    ignore_file = src / ".gitignore"
    ignore_file = ignore_file if ignore_file.exists() else None
    copy_files(app.local_app_dir, dest, ignore_file)
    potential_requirements_file = dest / "requirements.txt"
    requirements = build_proper_requirements(potential_requirements_file)

    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name) as run:
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=app,
            signature=app.signature(),
            artifacts={app.code_key: str(dest)},
            pip_requirements=requirements
        )

    mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", uc_model_path)
    for alias in aliases:
        ws_client.registered_models.set_alias(
            full_name=uc_model_path,
            version_num=mv.version,
            alias=alias
        )
    return mv


def keep_only_last_n_versions(ws_client: WorkspaceClient, uc_model_path: str, n: int):
    versions = ws_client.model_versions.list(uc_model_path)
    versions = sorted(versions, key=lambda x: x.version, reverse=True)
    # delete any versions other than the last n versions
    if len(versions) > n:
        for version in versions[n:]:
            print("Deleting version", version.version)
            ws_client.model_versions.delete(uc_model_path, version.version)


def copy_files(src, dest, ignore_file=None):
    spec = None
    src_dir = str(src)
    dest_dir = str(dest)
    if ignore_file is not None:
        ignore_file = str(ignore_file)
        with open(ignore_file, 'r') as f:
            gitignore = f.read()
        spec = PathSpec.from_lines(GitWildMatchPattern, gitignore.splitlines())

    # clean the destination directory
    if dest.exists():
        shutil.rmtree(dest)

    for root, dirs, files in os.walk(src_dir):
        for directory in dirs:
            dest_subdir = dest_dir / Path(root).relative_to(src_dir) / directory
            dest_subdir.mkdir(parents=True, exist_ok=True)

        for file in files:
            file_path = str(os.path.join(root, file))
            dest_file_path = str(dest_dir / Path(root).relative_to(src_dir) / file)
            if ignore_file is None:
                print("Copying", file_path, "to", dest_file_path)
                shutil.copy(file_path, dest_file_path)
                continue

            if not spec.match_file(file_path):
                print("Copying", file_path, "to", dest_file_path)
                shutil.copy(file_path, dest_file_path)


def deploy_serving_endpoint(ws_client: WorkspaceClient, endpoint_name: str, uc_model_path: str,
                            model_version: ModelVersion):
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
