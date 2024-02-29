import os
import tempfile
from pathlib import Path
from subprocess import CalledProcessError
from typing import Dict, Optional
from urllib.parse import urlparse

import click
from databricks.sdk import WorkspaceClient
from dotenv import dotenv_values

from mlrpc.cfg import ConfigFileProcessor
from mlrpc.deployment import get_or_create_mlflow_experiment, save_model, keep_only_last_n_versions
from mlrpc.flavor import FastAPIFlavor, pack_env_file_into_preload
from mlrpc.utils import execute, find_next_open_port, get_profile_contents, DatabricksProfile


@click.group()
def cli():
    pass


def configure_mlflow_to_databricks(env_dict: Dict[str, str], profile: Optional[DatabricksProfile] = None):
    env_dict["MLFLOW_TRACKING_URI"] = "databricks"
    env_dict["MLFLOW_REGISTRY_URI"] = "databricks-uc"
    if profile is not None:
        env_dict["DATABRICKS_HOST"] = profile.host
        env_dict["DATABRICKS_TOKEN"] = profile.token
    return env_dict


def serve_mlflow_model_cmd(model_uri: str, port: int):
    return [
        "mlflow",
        "models",
        "serve",
        "-h",
        "0.0.0.0",
        "-m",
        model_uri,
        "-p",
        str(port)
    ]


def get_only_host(url: str):
    return urlparse(url).hostname


def get_catalog_url(host: str, uc_name: str, version: str):
    catalog, schema, model = uc_name.split(".")
    return f"https://{host}/explore/data/models/{catalog}/{schema}/{model}/version/{version}"


def get_experiment_url(host: str, experiment_id: str):
    return f"https://{host}/ml/experiments/{experiment_id}"


CONTEXT_SETTINGS = dict(default_map=ConfigFileProcessor.read_config())


def ensure_run_or_uc(run_name: str, catalog, schema, name, latest_alias_name):
    if run_name is None and (catalog is None or schema is None or name is None or latest_alias_name is None):
        raise click.ClickException("Either provide a run name or a catalog, schema and model name")
    if run_name is not None and (
            catalog is not None or schema is not None or name is not None or latest_alias_name is not None):
        raise click.ClickException("Provide either a run name or a catalog, schema and model name")


def ensure_databrickscfg_exists():
    if not Path("~/.databrickscfg").expanduser().exists():
        raise click.ClickException("No databrickscfg file found in ~/.databrickscfg. Please make a profile")

@cli.command(context_settings=CONTEXT_SETTINGS)
@click.option("-c", "--catalog", "uc_catalog", type=str)
@click.option("-s", "--schema", "uc_schema", type=str)
@click.option("-n", "--name", "name", type=str)
@click.option("-a", "--alias", "latest_alias_name", type=str)
@click.option("-r", "--run-name", "run_name", type=str)
@click.option("-p", "--profile", "databricks_profile", type=str, default="DEFAULT")
@click.option("-e", "--env-file", "envfile", type=click.Path(exists=True, resolve_path=True, file_okay=True), default=None)
@click.pass_context
def local(
        ctx,
        uc_catalog: str,
        uc_schema: str,
        name: str,
        run_name: str,
        latest_alias_name: str,
        databricks_profile: str,
        envfile: str
):
    ensure_run_or_uc(run_name, uc_catalog, uc_schema, name, latest_alias_name)
    ensure_databrickscfg_exists()
    profile = get_profile_contents(databricks_profile)
    env_copy = os.environ.copy()
    env_copy = configure_mlflow_to_databricks(env_copy, profile)
    port = find_next_open_port()
    ws = WorkspaceClient(profile=databricks_profile)
    # TODO: this is a hack with some weird boto3 bug if you try to directly access the model version
    if envfile is not None:
        pack_env_file_into_preload(Path(envfile), env_copy)
    uc_name = f"{uc_catalog}.{uc_schema}.{name}"
    alias = latest_alias_name
    v = ws.model_versions.get_by_alias(uc_name, alias)
    host = get_only_host(profile.host)
    click.echo(click.style(f"Model URL: {get_catalog_url(host, uc_name, str(v.version))}", fg="green"))
    click.echo("\n")
    try:
        for log in execute(
                cmd=serve_mlflow_model_cmd(
                    # "models:/srituc.models.demo_app@current"
                    f"runs:/{v.run_id}/model"
                    , port),
                env=env_copy,
        ):
            click.echo(log)
    except CalledProcessError as e:
        raise click.ClickException("Error serving model")


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.option("-c", "--catalog", "uc_catalog", type=str)
@click.option("-s", "--schema", "uc_schema", type=str)
@click.option("--app-root-dir", "app_root_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--app-path-in-root-dir", "app_path_in_root", type=str)
@click.option("--app-object", "app_obj", type=str)
@click.option("-n", "--name", "name", type=str)
@click.option("-a", "--alias", "latest_alias_name", type=str)
@click.option("--make-experiment", "make_experiment", type=bool, default=True)
@click.option("--experiment-name", "experiment_name", type=str, default=None)
@click.option("-r", "--register-model", "register_model", type=bool, default=True)
@click.option("-p", "--databricks-profile", "databricks_profile", type=str, default="default")
@click.option("-e", "--env", "env", type=str, default=None)
@click.option("--only-last-n-versions", "only_last_n_versions", type=int, default=None)
@click.pass_context
def deploy(ctx, *,
           uc_catalog: str,
           uc_schema: str,
           app_root_dir: str,
           app_path_in_root: str,
           app_obj: str,
           name: str,
           latest_alias_name: str,
           make_experiment: bool,
           experiment_name: str,
           register_model: bool,
           databricks_profile: str,
           only_last_n_versions: int,
           env: str
           ):
    ensure_databrickscfg_exists()
    if databricks_profile is not None:
        profile = get_profile_contents(databricks_profile)
        configure_mlflow_to_databricks(os.environ, profile)
        host = get_only_host(profile.host)
    else:
        host = get_only_host(os.environ["DATABRICKS_HOST"])

    ws = WorkspaceClient(profile=databricks_profile)
    uc_name = f"{uc_catalog}.{uc_schema}.{name}"
    generated_experiment_name = f"{uc_catalog}_{uc_schema}_{name}"
    model = FastAPIFlavor(local_app_dir_abs=str(app_root_dir),
                          local_app_path_in_dir=app_path_in_root,
                          app_obj=app_obj)
    if make_experiment is True:
        created_experiment_name = generated_experiment_name if experiment_name is None else experiment_name
        exp = get_or_create_mlflow_experiment(ws, created_experiment_name)
        click.echo(click.style(f"Created experiment {created_experiment_name}", fg="green"))
        click.echo(click.style(f"Experiment URL: {get_experiment_url(host, exp.experiment_id)}", fg="green"))
    else:
        created_experiment_name = generated_experiment_name if experiment_name is None else experiment_name
        exp = ws.experiments.get_by_name(created_experiment_name)
        click.echo(click.style(f"Using experiment {uc_name}", fg="green"))
        click.echo(click.style(f"Experiment URL: {get_experiment_url(host, exp.experiment.experiment_id)}", fg="green"))

    if register_model is True:
        click.echo(click.style("Registering model", fg="green"))
        with tempfile.TemporaryDirectory() as temp_dir:
            model_version = save_model(ws_client=ws,
                                       experiment=exp,
                                       app=model,
                                       uc_model_path=uc_name,
                                       dest_path=str(temp_dir),
                                       aliases=[latest_alias_name])
            click.echo(
                click.style(f"Model URL: {get_catalog_url(host, uc_name, str(model_version.version))}", fg="green"))
            if only_last_n_versions is not None and only_last_n_versions > 1:
                click.echo(click.style(f"Only last {only_last_n_versions} versions will be kept", fg="green"))
                keep_only_last_n_versions(ws, uc_name, only_last_n_versions)
    else:
        click.echo(click.style("Skipping model registration", fg="yellow"))
