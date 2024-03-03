import os
import shutil
from importlib.metadata import distributions
from pathlib import Path
from typing import Optional, List, Literal

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import ModelVersionInfo
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput, EndpointStateReady
from mlflow.entities import Experiment
from mlflow.entities.model_registry import ModelVersion
from pathspec import PathSpec
from pathspec.patterns import GitWildMatchPattern

from mlrpc.flavor import FastAPIFlavor, MLRPC_ENV_VARS_PRELOAD_KEY
from mlrpc.utils import get_version


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


class LibraryManager:

    def __init__(self):
        self._installed_libraries = {
            i.metadata['name']: i.version for i in distributions()
        }

    def has_library(self, library_name) -> bool:
        return library_name in self._installed_libraries

    def get_library_version(self, library_name):
        return self._installed_libraries.get(library_name)

    def library_pinned_string(self, library_name):
        return f"{library_name}=={self.get_library_version(library_name)}"


def default_mlrpc_libs():
    lbm = LibraryManager()
    libs = []
    important_libs = ["cloudpickle", "mlflow-skinny", "fastapi", "pandas", "httpx", "python-dotenv"]
    mlrpc_req = "mlrpc"
    mlflow_req = "mlflow"
    if lbm.has_library(mlrpc_req) is False or len(lbm.get_library_version(mlrpc_req).split(".")) > 3:
        libs.append("mlrpc")
    else:
        libs.append(lbm.library_pinned_string(mlrpc_req))

    if lbm.has_library(mlflow_req) is False:
        if lbm.has_library("mlflow-skinny") is True:
            libs.append(lbm.library_pinned_string("mlflow-skinny"))
            libs.append(lbm.library_pinned_string("mlflow-skinny").replace("mlflow-skinny", "mlflow"))
        else:
            libs.append("mlflow")
    else:
        libs.append(lbm.library_pinned_string(mlflow_req))

    for lib in important_libs:
        if lbm.has_library(lib) is False:
            libs.append(lib)
        else:
            libs.append(lbm.library_pinned_string(lib))

    return libs


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


def stage_files_for_deployment(app: FastAPIFlavor, dest_path: Optional[str] = None):
    src = Path(app.local_app_dir)
    dest = Path(dest_path)
    ignore_file = src / ".gitignore"
    ignore_file = ignore_file if ignore_file.exists() else None
    copy_files(app.local_app_dir, dest, ignore_file)


def save_model(
        ws_client: WorkspaceClient,
        experiment: Experiment,
        app: FastAPIFlavor,
        uc_model_path: str,
        run_name: str = "deployment",
        code_path: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        latest_alias_name: Optional[str] = "current",
        reload: bool = False,
) -> ModelVersion:
    if not ensure_3_parts(uc_model_path):
        raise ValueError \
            (f"Model path must be in the format 'catalog_name.schema_name.model_name' but got {uc_model_path}")

    aliases = aliases or []
    aliases.append(latest_alias_name)
    code_path = Path(code_path)
    potential_requirements_file = code_path / "requirements.txt"
    requirements = build_proper_requirements(potential_requirements_file)

    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name) as run:
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=app,
            signature=app.signature(),
            artifacts={app.code_key: str(code_path)},
            pip_requirements=requirements
        )
        mlflow.set_tag("mlrpc_version", get_version("mlrpc"))
        mlflow.set_tag("reloadable", 'true' if reload else 'false')

    mv = mlflow.register_model(f"runs:/{run.info.run_id}/model",
                               uc_model_path,
                               tags={
                                   "mlrpc_version": get_version("mlrpc"),
                                   "reloadable": 'true' if reload else 'false'
                               })
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


def deploy_secret_env_file(ws_client: WorkspaceClient, secret_scope: str, secret_key: str, env_file: Path):
    if env_file.exists() is False:
        return
    with open(env_file, 'r') as file:
        string_value = file.read()
        ws_client.secrets.put_secret(secret_scope, secret_key, string_value=string_value)


def _check_deployable(ws_client: WorkspaceClient, endpoint_name: str) -> Literal["DEPLOY", "UPDATE", "NOT_UPDATABLE"]:
    try:
        endpoint = ws_client.serving_endpoints.get(endpoint_name)
        # return endpoint.state.ready == EndpointStateReady.READY
        if endpoint.state.ready == EndpointStateReady.READY:
            return "UPDATE"
        return "NOT_UPDATABLE"
    except Exception:
        return "DEPLOY"


def deploy_serving_endpoint(ws_client: WorkspaceClient,
                            endpoint_name: str,
                            uc_model_path: str,
                            model_version: ModelVersion | ModelVersionInfo,
                            secret_scope: Optional[str] = None,
                            secret_key: Optional[str] = None,
                            size: Literal["Small", "Medium", "Large"] = "Small",
                            scale_to_zero_enabled: bool = True,
                            ):
    if _check_deployable(ws_client, endpoint_name) == "NOT_UPDATABLE":
        raise ValueError(f"Endpoint {endpoint_name} is not ready state to be updated")

    if size not in ["Small", "Medium", "Large"]:
        raise ValueError(f"Size must be one of 'Small', 'Medium', 'Large' but got {size}")

    env_vars = None
    if secret_scope is not None and secret_key is not None:
        env_vars = {
            MLRPC_ENV_VARS_PRELOAD_KEY: "{{" + f"secrets/{secret_scope}/{secret_key}" + "}}"
        }

    if _check_deployable(ws_client, endpoint_name) == "UPDATE":
        return ws_client.serving_endpoints.update_config_and_wait(
            name=endpoint_name,
            served_entities=[
                ServedEntityInput(
                    entity_version=model_version.version,
                    entity_name=uc_model_path,
                    workload_size=size,
                    scale_to_zero_enabled=scale_to_zero_enabled,
                    environment_vars=env_vars
                )
            ],
        )

    return ws_client.serving_endpoints.create_and_wait(
        name=endpoint_name,
        config=EndpointCoreConfigInput(
            name=endpoint_name,
            served_entities=[
                ServedEntityInput(
                    entity_version=model_version.version,
                    entity_name=uc_model_path,
                    workload_size=size,
                    scale_to_zero_enabled=scale_to_zero_enabled,
                    environment_vars=env_vars
                )
            ],
        )
    )
