import base64
import json
from dataclasses import dataclass
from typing import Literal, Sequence, Tuple, Optional, List

import httpx
import mlflow
import pandas as pd


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
    def from_httpx_resp(cls, resp: httpx.Response):
        return cls(
            status_code=resp.status_code,
            headers=list(resp.headers.items()),
            content=resp.json()
        )

    @classmethod
    def from_mlflow_predict(cls, input: pd.DataFrame) -> List['ResponseObject']:
        response_objs_enc = [ResponseObjectEncoded(**row) for row in input.to_dict(orient='records')]
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


class FastAPIFlavor(mlflow.pyfunc.PythonModel):

    def __init__(self,
                 *,
                 local_app_dir_abs: Optional[str],
                 local_app_path_in_dir: Optional[str] = "app.py",
                 app_obj: Optional[str] = "app",
                 artifact_code_key="code"):
        self.local_app_dir = local_app_dir_abs
        self.code_key = artifact_code_key
        self.local_app_dir = self.local_app_dir.rstrip("/")
        self.app_path_in_dir = local_app_path_in_dir
        self.app_obj = app_obj
        self._app = None
        self._client = None
        self.validate()

    def load_module(self, app_dir_mlflow_artifacts: Optional[str] = None):
        import site
        app_dir = app_dir_mlflow_artifacts or self.local_app_dir
        # only add if it's not already there
        site.addsitedir(app_dir)
        app_module = __import__(self.app_path_in_dir.replace('.py', '').replace('/', '.'), fromlist=[self.app_obj])
        print(f"Loaded module successfully for {self.app_path_in_dir} from {self.local_app_dir}")
        app_obj = getattr(app_module, self.app_obj)
        print(f"Loaded app object successfully for {self.app_obj}")
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
        from fastapi.testclient import TestClient

        code_path = None
        if context is not None:
            code_path = context.artifacts[self._code_key]
        # add the app_path to the pythonpath to load modules
        # only if it's not already there
        # update site packages to include this app dir
        self._app = self.load_module(app_dir_mlflow_artifacts=code_path)
        self._client = TestClient(self._app)

    def predict(self, context, model_input: pd.DataFrame, params=None):
        if self._app is None:
            self.load_context(context)

        requests = RequestObject.from_encoded_df(model_input)
        responses = []
        for req in requests:
            resp = self._client.request(
                method=req.method,
                url=req.path,
                headers=req.headers,
                params=req.query_params,
                data=req.content,
                timeout=req.timeout
            )
            responses.append(ResponseObject.from_httpx_resp(resp).encode())
        return pd.DataFrame([resp.dict() for resp in responses])

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
