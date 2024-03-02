import hashlib
import io
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Dict, List

import mlflow
from mlflow.models import infer_signature
import pandas as pd
from dotenv import dotenv_values
from starlette.testclient import TestClient

from mlrpc.proto import base64_to_dir, ResponseObject, RequestObject, RequestObjectEncoded

MLRPC_ENV_VARS_PRELOAD_KEY = "MLRPC_ENV_VARS_PRELOAD"


def request_to_df(request: RequestObject):
    return pd.DataFrame([request.encode().dict()])


def response_to_df(response: ResponseObject):
    return pd.DataFrame([response.encode().dict()])


def make_request_from_input_df(input_df: pd.DataFrame) -> List['RequestObject']:
    request_objs_enc = [RequestObjectEncoded(**row) for row in input_df.to_dict(orient='records')]
    return [RequestObject.from_request_enc(enc) for enc in request_objs_enc]


def load_mlrpc_env_vars() -> None:
    env_vars = os.getenv(MLRPC_ENV_VARS_PRELOAD_KEY, "")
    buff = io.StringIO(env_vars)
    loaded_env_vars = dotenv_values(stream=buff)
    for k, v in loaded_env_vars.items():
        os.environ[k] = v


def pack_env_file_into_preload(envfile: Path, env_dict: Dict[str, str] = None):
    env_dict = env_dict or os.environ
    if envfile.exists() is True:
        with envfile.open("r") as f:
            env_vars = f.read()
        env_dict[MLRPC_ENV_VARS_PRELOAD_KEY] = env_vars


def copy_files(src: Path, dest: Path, check_dest_empty: bool = True):
    # copying due to not wanting requirements for pathspec
    src_dir = str(src)
    dest_dir = str(dest)

    if dest.exists() and any(dest.iterdir()) and check_dest_empty is True:
        print("Destination directory is not empty. Skipping file copy.")
        return

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
            print("Copying", file_path, "to", dest_file_path, flush=True)
            shutil.copy(file_path, dest_file_path)


class AppClientProxy:

    def __init__(self, app):
        self._app = app
        self._client = TestClient(app)

    @property
    def app(self):
        return self._app

    @property
    def client(self):
        return self._client

    def update_app(self, app):
        self._app = app
        self._client = TestClient(app)


class HotReloadEventHandler:
    PATH_PREFIX = "__INTERNAL__"
    VALID_EVENTS = ["FULL_SYNC", "RELOAD", "REINSTALL", "RESET"]

    def __init__(self,
                 *,
                 app_client_proxy: AppClientProxy,
                 reload_code_path: str,
                 file_in_code_path: str,
                 obj_name: str,
                 reset_code_path: Optional[str] = None):
        self._reset_code_path = reset_code_path
        self._obj_name = obj_name
        self._file_in_code_path = file_in_code_path
        self._app_client_proxy = app_client_proxy
        self._reload_code_path = reload_code_path

    def validate(self):
        print(
            f"Validating hot reload dispatcher with {self._app_client_proxy.app} {self._reload_code_path} {self._file_in_code_path} {self._obj_name}",
            flush=True)
        if not self._app_client_proxy.app:
            raise ValueError("app_proxy must be set")
        if not self._reload_code_path:
            raise ValueError("code_path must be set")
        if not self._file_in_code_path:
            raise ValueError("file_in_code_path must be set")
        if not self._obj_name:
            raise ValueError("obj_name must be set")

    def _is_valid_event(self, path: str) -> bool:
        normalized_req_path = path.lstrip("/").rstrip("/")
        valid_events = [f"{self.PATH_PREFIX}/{event}" for event in self.VALID_EVENTS]
        return normalized_req_path in valid_events

    def reload_app(self):
        import site
        import importlib
        site.addsitedir(self._reload_code_path)
        app_module = __import__(self._file_in_code_path.replace('.py', '').replace('/', '.'),
                                fromlist=[self._obj_name])
        importlib.reload(app_module)
        app_obj = getattr(app_module, self._obj_name)
        self._app_client_proxy.update_app(app_obj)
        print("Reloaded App!", flush=True)

    def _validate_checksum(self, content: str, checksum: str) -> bool:
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        print("Valid checksum", flush=True)
        return content_hash == checksum

    def _do_full_sync(self, request: RequestObject):
        payload = json.loads(request.content)
        content = payload['content']
        checksum = payload['checksum']
        if self._validate_checksum(content, checksum) is False:
            return ResponseObject(
                status_code=400,
                content="Checksum validation failed"
            )
        base64_to_dir(content, self._reload_code_path)
        # reload after files get moved
        self.reload_app()
        return ResponseObject(
            status_code=200,
            content="SUCCESS"
        )

    @staticmethod
    def _install_packages(package_names: List[str]):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *package_names])
            return f"Packages {','.join(package_names)} installed successfully"
        except subprocess.CalledProcessError:
            raise ValueError(f"Failed to install packages {','.join(package_names)}")

    @staticmethod
    def _uninstall_package(package_name: str):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", package_name])
            return f"Package {package_name} uninstalled successfully"
        except subprocess.CalledProcessError:
            f"Failed to uninstall package {package_name}"

    def _do_install(self, request: RequestObject):
        payload = json.loads(request.content)
        requirements = payload['requirements']
        install_message = self._install_packages(requirements)
        return ResponseObject(
            status_code=200,
            content=json.dumps(install_message)
        )

    def dispatch(self, request: RequestObject) -> Optional[ResponseObject]:
        if request.method != "POST":
            return None

        if self._is_valid_event(request.path) is False:
            return None

        if request.path == f"/{self.PATH_PREFIX}/FULL_SYNC":
            print("Dispatching full sync event", flush=True)
            self._do_full_sync(request)
            return ResponseObject(
                status_code=200,
                content="SUCCESS"
            )

        if request.path == f"/{self.PATH_PREFIX}/REINSTALL":
            print("Dispatching reinstall event", flush=True)
            return self._do_install(request)


class FastAPIFlavor(mlflow.pyfunc.PythonModel):

    def __init__(self,
                 *,
                 local_app_dir_abs: Optional[str],
                 local_app_path_in_dir: Optional[str] = "app.py",
                 app_obj: Optional[str] = "app",
                 artifact_code_key="code",
                 reloadable: Optional[bool] = False):
        self._reloadable = reloadable
        self.local_app_dir = local_app_dir_abs
        self.code_key = artifact_code_key
        self.local_app_dir = self.local_app_dir.rstrip("/")
        self.app_path_in_dir = local_app_path_in_dir
        self.app_obj = app_obj
        self._app_proxy = None
        self._hot_reload_dispatcher: Optional[HotReloadEventHandler] = None

    def load_module(self, app_dir_mlflow_artifacts: Optional[str] = None):
        import site
        print("Loading preloaded env vars", flush=True)
        app_dir = app_dir_mlflow_artifacts or self.local_app_dir
        # only add if it's not already there
        site.addsitedir(app_dir)
        print("Loading code from path", app_dir, flush=True)
        app_module = __import__(self.app_path_in_dir.replace('.py', '').replace('/', '.'), fromlist=[self.app_obj])
        print(f"Loaded module successfully for {self.app_path_in_dir} from {app_dir}", flush=True)
        app_obj = getattr(app_module, self.app_obj)
        print(f"Loaded app object successfully for {self.app_obj}", flush=True)
        return app_obj

    def validate(self):
        if not self.local_app_dir:
            raise ValueError("local_app_dir_abs must be set")
        if not self.app_path_in_dir:
            raise ValueError("app_path_in_dir must be set")
        if not self.app_obj:
            raise ValueError("app_obj must be set")
        try:
            self.load_module()
        except Exception as e:
            raise ValueError(f"Failed to load app module: {e} in {self.local_app_dir}/{self.app_path_in_dir}")

    def load_context(self, context):
        load_mlrpc_env_vars()
        code_path = self.local_app_dir
        if context is not None:
            code_path = context.artifacts[self.code_key]

        self._app_proxy = AppClientProxy(None)

        # add the app_path to the pythonpath to load modules
        # only if it's not already there
        # update site packages to include this app dir

        def boot_app():
            Path(temp_code_dir).mkdir(parents=True, exist_ok=True)
            print("Hot reload dir being created at /tmp/mlrpc-hot-reload", flush=True)

            copy_files(Path(code_path), Path(temp_code_dir), check_dest_empty=True)
            self._hot_reload_dispatcher = HotReloadEventHandler(
                app_client_proxy=self._app_proxy,
                reload_code_path=temp_code_dir,
                file_in_code_path=self.app_path_in_dir,
                obj_name=self.app_obj
            )
            self._app_proxy.update_app(self.load_module(app_dir_mlflow_artifacts=temp_code_dir))

        if os.getenv("MLRPC_HOT_RELOAD", str(self._reloadable)).lower() == "true":
            temp_code_dir = "/tmp/mlrpc-hot-reload"
            attempts = 5
            while True:
                try:
                    boot_app()
                    break
                except Exception as e:
                    print(f"Error occurred while booting app: {e}", flush=True)
                    attempts -= 1
                    wait_time = random.uniform(1, 5)
                    time.sleep(wait_time)
                    if attempts == 0:
                        raise
                    print(f"Retrying in 5 seconds", flush=True)
        else:
            self._app_proxy.update_app(self.load_module(app_dir_mlflow_artifacts=code_path))

    def predict(self, context, model_input: pd.DataFrame, params=None):
        if self._app_proxy is None:
            self._app_proxy = AppClientProxy(None)

        if self._app_proxy.app is None or self._app_proxy.client is None:
            self.load_context(context)

        requests = make_request_from_input_df(model_input)
        responses = []

        try:
            # happy path things can go wrong :-)
            if self._hot_reload_dispatcher is not None and len(requests) == 1:
                print("Checking for hot reload events", flush=True)
                event_resp = self._hot_reload_dispatcher.dispatch(requests[0])
                if event_resp is not None:
                    return response_to_df(event_resp)

            if self._hot_reload_dispatcher is not None:
                self._hot_reload_dispatcher.reload_app()

            for req in requests:
                resp = self._app_proxy.client.request(
                    method=req.method,
                    url=req.path,
                    headers=req.headers,
                    params=req.query_params,
                    data=req.content,
                    timeout=req.timeout
                )
                responses.append(ResponseObject.from_httpx_resp(resp).encode())
            return pd.DataFrame([resp.dict() for resp in responses])
        except Exception as e:
            return response_to_df(ResponseObject(
                status_code=500,
                content=f"Error occurred: {str(e)}"
            ))

    def signature(self):
        full_request = request_to_df(RequestObject(
            method='GET',
            headers=[('Host', 'example.org')],
            path='/',
            query_params="",
            content="",
            timeout=10
        ))
        optional_request = request_to_df(RequestObject(
            method='GET',
            path='/'
        ))
        full_response = response_to_df(ResponseObject(
            status_code=200,
            headers=[('Content-Type', 'application/json')],
            content=""
        ))
        optional_response = response_to_df(ResponseObject(
            status_code=200,
        ))
        return infer_signature(
            model_input=pd.concat([full_request, optional_request]),
            model_output=pd.concat([full_response, optional_response])
        )
