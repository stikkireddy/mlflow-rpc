import os

from click_configfile import ConfigFileReader, Param, SectionSchema
from click_configfile import matches_section
import click


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


class ConfigFileProcessor(ConfigFileReader):
    config_files = ["mlrpc.ini", "mlrpc.cfg"]
    config_section_schemas = [
        ConfigSectionSchema.Root,     # PRIMARY SCHEMA
    ]
