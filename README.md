# mlrpc [Experimental]

Host REST APIs via FastAPI in Databricks Serverless Model serving via this rpc abstraction.

MLRPC does not modify your FastAPI code! It acts as a build step/proxy layer
and invokes your FastAPI code in a databricks serverless model serving environment and routes to the right endpoints 
properly without you having to know about custom python models. 

This is currently experimental in nature. If interest increases, will work on hardening and ensure it has a stable build
and tests.

## Key features

1. **FastAPI**: Host your FastAPI code in databricks
2. Generate OpenAPI client objects
3. Swagger UI to explore your endpoint
4. Hot reload your FastAPI code to the remote endpoint
5. Hot reloads of code are encrypted
6. No need to manage your own pyfunc model
7. Full support for environments (dev, test, prod) and interactive development

## Limitations

1. Only supports FastAPI
2. Requires mlrpc client or follow spec to query the endpoint
3. Does not support FastAPI lifecycle events

## Installation

Use the cli to install the package

```bash
pip install -U 'mlrpc[cli]'
```

## Instructions

Keep in mind that the cli has a lot of options but all of them are also managable via the config file.

### 1. Make a config file

```
mlrpc init
```

### 2. Edit the config file

```toml
## The following lines are minimum required config
## You need to have the app section
# [app]
# name = <app name>
# uc_catalog = <catalog name>
# uc_schema = <schema name>
# endpoint_name = <endpoint name>

# data_dir=data # this is optional if you want to upload larger binaries like chroma, sqlite, faiss, lancedb, etc

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

This may not work for you if your models require gpus or other resources that wont exist on your laptop

```
mlrpc local -p <databricks profile>
```

### 5. Deploy to model serving infra

```
mlrpc serve -p <databricks profile>
```

### 6. Explore the deployed endpoint via swagger proxy and hot reloading

```
mlrpc swagger -p <databricks profile>
```

## Disclaimer
mlrpc is not developed, endorsed not supported by Databricks. It is provided as-is; no warranty is derived from using this package. 
For more details, please refer to the license.


