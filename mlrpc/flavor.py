import base64
import hashlib
import io
import json
import os
import random
import shutil
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence, Tuple, Optional, List, Dict

import httpx
import mlflow
import pandas as pd
from dotenv import dotenv_values
from starlette.testclient import TestClient


def _b64encode(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    return base64.b64encode(s.encode('utf-8')).decode('utf-8')


def _b64decode(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    return base64.b64decode(s).decode('utf-8')


def _encode_anything(s: any) -> Optional[str]:
    if s is None:
        return None
    if isinstance(s, bytes):
        s = s.decode('utf-8')

    json_str = json.dumps(s)
    return _b64encode(json_str)


def _decode_anything(s: Optional[str]) -> any:
    if s is None:
        return None
    json_str = _b64decode(s)
    return json.loads(json_str)


def base64_to_dir(base64_string, target_dir):
    # Decode the base64 string back to bytes
    base64_data = base64.b64decode(base64_string)

    # Create a BytesIO object from these bytes
    data = io.BytesIO(base64_data)

    # Remove all files in the target directory
    shutil.rmtree(target_dir)
    os.makedirs(target_dir)

    # Open the tarfile for reading from the binary stream
    with tarfile.open(fileobj=data, mode='r:gz') as tar:
        # Extract all files to the target directory
        tar.extractall(path=target_dir)


@dataclass
class ResponseObjectEncoded:
    status_code: int
    headers: Optional[str] = None
    content: Optional[str] = None

    def dict(self):
        return {
            "status_code": self.status_code,
            "headers": self.headers,
            "content": self.content
        }

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame([self.dict()])

    def to_mlflow_df_split_dict(self):
        return {
            "dataframe_split": {
                "columns": ["status_code", "headers", "content"],
                "data": [[self.status_code, self.headers, self.content]],
            }
        }

    def to_mlflow_df_split_json(self):
        return json.dumps(self.to_mlflow_df_split_dict())


@dataclass
class ResponseObject:
    status_code: int
    headers: Optional[Sequence[Tuple[str, str]]] = None
    content: Optional[str] = None

    def encode(self):
        headers = [{header[0]: header[1]} for header in self.headers] if self.headers is not None else None
        return ResponseObjectEncoded(
            status_code=self.status_code,
            headers=_encode_anything(headers),
            content=_encode_anything(self.content)
        )

    @classmethod
    def from_resp_enc(cls, encoded: ResponseObjectEncoded):
        status_code = encoded.status_code
        headers = _decode_anything(encoded.headers) if encoded.headers is not None else []
        headers = [(k, v) for header in headers for k, v in header.items()]
        content = _decode_anything(encoded.content)
        return cls(
            status_code=status_code,
            headers=headers,
            content=content
        )

    @classmethod
    def from_serving_resp(cls, resp: Dict[str, str | int]):
        return cls(
            status_code=resp['status_code'],
            headers=_decode_anything(resp['headers']),
            content=_decode_anything(resp['content'])
        )

    @classmethod
    def from_httpx_resp(cls, resp: httpx.Response):
        # TODO: being lazy here, should handle more gracefully using mimetype
        try:
            return cls(
                status_code=resp.status_code,
                headers=list(resp.headers.items()),
                content=resp.json()
            )
        except Exception as e:
            return cls(
                status_code=resp.status_code,
                headers=list(resp.headers.items()),
                content=resp.text
            )

    @classmethod
    def from_mlflow_predict(cls, _input: pd.DataFrame) -> List['ResponseObject']:
        response_objs_enc = [ResponseObjectEncoded(**row) for row in _input.to_dict(orient='records')]
        return [cls.from_resp_enc(enc) for enc in response_objs_enc]


@dataclass
class RequestObjectEncoded:
    method: str
    path: str
    headers: Optional[str] = None
    query_params: Optional[str] = None
    content: Optional[str] = None
    timeout: Optional[float] = None

    def dict(self):
        return {
            "method": self.method,
            "path": self.path,
            "headers": self.headers,
            "query_params": self.query_params,
            "content": self.content,
            "timeout": self.timeout
        }

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame([self.dict()])

    def to_mlflow_df_split_dict(self):
        return {
            "dataframe_split": {
                "columns": ["method", "headers", "path", "query_params", "content", "timeout"],
                "data": [[self.method, self.headers, self.path, self.query_params, self.content, self.timeout]],
            }
        }

    def to_sdk_df_split(self):
        return {
            "columns": ["method", "headers", "path", "query_params", "content", "timeout"],
            "data": [[self.method, self.headers, self.path, self.query_params, self.content, self.timeout]],
        }

    def to_mlflow_df_split_json(self):
        return json.dumps(self.to_mlflow_df_split_dict())


@dataclass
class RequestObject:
    method: Literal['GET', 'POST', 'PUT', 'DELETE', 'PATCH']
    path: str
    headers: Optional[Sequence[Tuple[str, str]]] = None
    query_params: Optional[str] = None  # todo make this a bit more typed in future
    content: Optional[str] = None
    timeout: Optional[float] = None

    def encode(self):
        headers = [{header[0]: header[1]} for header in self.headers] if self.headers is not None else None
        return RequestObjectEncoded(
            method=self.method,
            headers=_encode_anything(headers),
            path=self.path,
            query_params=self.query_params,
            content=_encode_anything(self.content),
            timeout=self.timeout
        )

    @classmethod
    def from_request_enc(cls, encoded: RequestObjectEncoded):
        method: Literal['GET', 'POST', 'PUT', 'DELETE', 'PATCH'] = encoded.method
        headers = _decode_anything(encoded.headers) if encoded.headers is not None else []
        headers = [(k, v) for header in headers for k, v in header.items()]
        path = encoded.path
        query_params = encoded.query_params
        content = _decode_anything(encoded.content)
        timeout = encoded.timeout
        return cls(
            method=method,
            headers=headers,
            path=path,
            query_params=query_params,
            content=content,
            timeout=timeout
        )

    @classmethod
    def from_encoded_df(cls, input: pd.DataFrame) -> List['RequestObject']:
        request_objs_enc = [RequestObjectEncoded(**row) for row in input.to_dict(orient='records')]
        return [cls.from_request_enc(enc) for enc in request_objs_enc]


MLRPC_ENV_VARS_PRELOAD_KEY = "MLRPC_ENV_VARS_PRELOAD"


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


class HotReloadEvents:

    @staticmethod
    def full_sync(content: str) -> RequestObject:
        return RequestObject(
            method="POST",
            path="/__INTERNAL__/FULL_SYNC",
            content=json.dumps({
                "content": content,
                "checksum": hashlib.md5(content.encode('utf-8')).hexdigest()
            })
        )

    @staticmethod
    def reload() -> RequestObject:
        return RequestObject(
            method="POST",
            path="/__INTERNAL__/RELOAD"
        )

    @staticmethod
    def reinstall() -> RequestObject:
        return RequestObject(
            method="POST",
            path="/__INTERNAL__/REINSTALL"
        )


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


class HotReloadEventDispatcher:
    PATH_PREFIX = "__INTERNAL__"
    VALID_EVENTS = ["FULL_SYNC", "RELOAD", "REINSTALL"]

    def __init__(self, app_proxy: AppClientProxy,
                 code_path: str,
                 file_in_code_path: str,
                 obj_name: str):
        self._obj_name = obj_name
        self._file_in_code_path = file_in_code_path
        self._app_proxy = app_proxy
        self._code_path = code_path
        # create temp dir to store the code
        # self.validate()

    def validate(self):
        print(
            f"Validating hot reload dispatcher with {self._app_proxy.app} {self._code_path} {self._file_in_code_path} {self._obj_name}",
            flush=True)
        if not self._app_proxy.app:
            raise ValueError("app_proxy must be set")
        if not self._code_path:
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
        site.addsitedir(self._code_path)
        app_module = __import__(self._file_in_code_path.replace('.py', '').replace('/', '.'),
                                fromlist=[self._obj_name])
        importlib.reload(app_module)
        app_obj = getattr(app_module, self._obj_name)
        self._app_proxy.update_app(app_obj)
        print("Reloaded App!", flush=True)

    def _validate_checksum(self, content: str, checksum: str) -> bool:
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        print("Valid checksum", flush=True)
        return content_hash == checksum

    def _do_full_sync(self, request: RequestObject):
        self.reload_app()
        payload = json.loads(request.content)
        content = payload['content']
        checksum = payload['checksum']
        if self._validate_checksum(content, checksum) is False:
            return ResponseObject(
                status_code=400,
                content="Checksum validation failed"
            )
        base64_to_dir(content, self._code_path)
        return ResponseObject(
            status_code=200,
            content="SUCCESS"
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
        self._hot_reload_dispatcher: Optional[HotReloadEventDispatcher] = None

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
            self._hot_reload_dispatcher = HotReloadEventDispatcher(
                self._app_proxy,
                temp_code_dir,
                self.app_path_in_dir,
                self.app_obj
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

        requests = RequestObject.from_encoded_df(model_input)
        responses = []

        try:
            # happy path things can go wrong :-)
            if self._hot_reload_dispatcher is not None and len(requests) == 1:
                print("Checking for hot reload events", flush=True)
                event_resp = self._hot_reload_dispatcher.dispatch(requests[0])
                if event_resp is not None:
                    return event_resp.encode().to_df()

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
            return ResponseObject(
                status_code=500,
                content=f"Error occurred: {str(e)}"
            ).encode().to_df()

    def signature(self):
        from mlflow.models import infer_signature
        full_request = RequestObject(
            method='GET',
            headers=[('Host', 'example.org')],
            path='/',
            query_params="",
            content="",
            timeout=10
        ).encode().to_df()
        optional_request = RequestObject(
            method='GET',
            path='/'
        ).encode().to_df()
        full_response = ResponseObject(
            status_code=200,
            headers=[('Content-Type', 'application/json')],
            content=""
        ).encode().to_df()
        optional_response = ResponseObject(
            status_code=200,
        ).encode().to_df()
        return infer_signature(
            model_input=pd.concat([full_request, optional_request]),
            model_output=pd.concat([full_response, optional_response])
        )
