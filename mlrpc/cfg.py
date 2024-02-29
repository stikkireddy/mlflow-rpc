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
        size = Param(type=str, default="Small")
        scale_to_zero_enabled = Param(type=bool, default=True)


class ConfigFileProcessor(ConfigFileReader):
    config_files = ["mlrpc.ini", "mlrpc.cfg"]
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

# Cost controls
# size = Small # Small, Medium, Large
# scale_to_zero_enabled = true # true, false

## The following lines are optional and not required

# latest_alias_name = <alternative alias> uses "current" by default

# experiment_name = <experiment name> # optional specify your experiment name

## Define where your code is
# app_root_dir = <root of your directory>

## The following lines are optional if you need secrets
# secret_scope = <secret scope in databricks>
# secret_key = <secret key>
# env_file = <location of your env file relative to this file>

## use this setting if you want to delete old versions
# only_last_n_versions=10
"""
