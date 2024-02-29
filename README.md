# mlflow rpc

Host REST APIs in Databricks Serverless Model serving via this rpc framework.

This framework offers you a restapi spec to be able to submit rpc calls and retrieve results.

## Installation

```bash
pip install mlrpc
```

## Instructions

Keep in mind that the cli has a lot of options but all of them are also managable via the config file.

### 1. Make a config file

```
mlrpc init
```

### 2. Edit the config file

```toml
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
```

### 3. Deploy the artifacts

```
mlrpc deploy -p <databricks profile>
```

### 4. [OPTIONAL] verify in valid model deployment server locally

```
mlrpc local -p <databricks profile>
```

### 5. Deploy to model serving infra

```
mlrpc serve -p <databricks profile>
```

### 6. Explore the deployed endpoint via swagger proxy

```
mlrpc serve -p <databricks profile>
```

