import os

import click
from click_configfile import ConfigFileReader, Param, SectionSchema
from click_configfile import matches_section


class ConfigSectionSchema(object):
    """Describes all config sections of this configuration file."""

    @matches_section("app")
    class Root(SectionSchema):
        name = Param(type=str)
        app_root_dir = Param(type=click.Path(), default=os.getcwd())
        app_path_in_root = Param(type=str, default="app.py")
        app_obj = Param(type=str, default="app")
        data_dir = Param(type=click.Path(exists=True, dir_okay=True), default=None)
        uc_catalog = Param(type=str)
        uc_schema = Param(type=str)
        endpoint_name = Param(type=str)
        latest_alias_name = Param(type=str, default="current")
        env = Param(type=str, default="dev")
        make_experiment = Param(type=bool, default=True)
        experiment_name = Param(type=str)
        register_model = Param(type=bool, default=True)
        databricks_profile = Param(type=str, default="default")
        only_last_n_versions = Param(type=int, default=None)
        secret_scope = Param(type=str, default=None)
        secret_key = Param(type=str, default=None)
        env_file = Param(type=click.Path(), default=None)
        type = Param(type=str, default="CPU")
        size = Param(type=str, default="Small")
        scale_to_zero_enabled = Param(type=bool, default=True)
        bootstrap_python_script = Param(type=click.Path(), default=None)


class ConfigFileProcessor(ConfigFileReader):
    config_files = ["mlrpc.ini", "mlrpc.cfg"] if os.getenv("MLRPC_CONFIG") is None else [os.getenv("MLRPC_CONFIG")]
    config_section_schemas = [
        ConfigSectionSchema.Root,  # PRIMARY SCHEMA
    ]


INIT_CONFIG = """## The following lines are minimum required config
## You need to have the app section
# [app]
# name = <app name>
# uc_catalog = <catalog name>
# uc_schema = <schema name>
# endpoint_name = <endpoint name>

# this is optional if you want to upload larger binaries like chroma, sqlite, faiss, lancedb, etc
# data_dir=data

# Cost controls
# Small, Medium, Large are valid values
# size = Small
# various values 
# refer to https://docs.databricks.com/en/machine-learning/model-serving/create-manage-serving-endpoints.html
# or https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/create-manage-serving-endpoints
# type = CPU
# scale_to_zero_enabled = true # true, false

# This is an optional field if you want to run a python script before the main app
# this is great if you want to spawn a vllm process, deepspeed-mi server, tei, tgi, some app in another lang.
# bootstrap_python_script = <path to your python script>

## The following lines are optional and not required

# latest_alias_name = <alternative alias> uses "current" by default

## optional specify your experiment name
# experiment_name = <experiment name> 

## Define where your code is
# app_root_dir = <root of your directory>

## The following lines are optional if you need secrets
# secret_scope = <secret scope in databricks>
# secret_key = <secret key>
# env_file = <location of your env file relative to this file>

## use this setting if you want to delete old versions
# only_last_n_versions=10
"""
