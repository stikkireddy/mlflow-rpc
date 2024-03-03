import abc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple, Literal, Dict, Union, List, Optional, Any, Type
from urllib.parse import urlencode

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import DataframeSplitInput

from mlrpc.proto import ResponseObject, RequestObject, HotReloadEvents, EncryptDecrypt

MethodType = Literal['GET', 'POST', 'PUT', 'DELETE', 'PATCH']
QueryParams = Dict[str, Union[str, List[str]]]
Headers = Sequence[Tuple[str, str]]
Body = Optional[str | Dict[str, Any] | List[Dict[str, Any]]]


@dataclass
class MLRPCResponse:
    request: RequestObject
    status_code: int
    headers: Optional[Sequence[Tuple[str, str]]] = None
    body: Optional[str | dict | list] = None

    @property
    def dict(self):
        if self.body is None:
            return {}
        if isinstance(self.body, dict):
            return self.body
        return json.loads(self.body)

    @property
    def json(self):
        return self.dict

    @property
    def text(self):
        if self.body is None:
            return ""
        if isinstance(self.body, str):
            return self.body
        return json.dumps(self.body)


def generate_query_string(params: QueryParams) -> str:
    return urlencode(params, doseq=True)


class DispatchHandler(abc.ABC):

    @abc.abstractmethod
    def dispatch(self, request: RequestObject) -> MLRPCResponse | List[MLRPCResponse]:
        pass


class MLRPCClient:
    def __init__(self, rpc_dispatch_handler: DispatchHandler):
        self._rpc_dispatch_handler: DispatchHandler = rpc_dispatch_handler

    def _dispatch(self,
                  *,
                  method: MethodType,
                  path: str,
                  query_params: Optional[QueryParams] = None,
                  data: Optional[Body] = None,
                  headers: Headers) -> MLRPCResponse:
        if query_params is not None:
            query_params = generate_query_string(query_params)
        if data is not None and isinstance(data, str) is False:
            data = json.dumps(data)
        request = RequestObject(
            method=method,
            path=path,
            query_params=query_params,
            headers=headers,
            content=data
        )
        resp = self._rpc_dispatch_handler.dispatch(request)
        if isinstance(resp, MLRPCResponse):
            return resp
        if isinstance(resp, list):
            return resp[0]

    def request(self,
                method: MethodType,
                path: str,
                query_params: Optional[QueryParams] = None,
                headers: Headers = None,
                data: Body = None,
                ) -> MLRPCResponse:
        return self._dispatch(method=method,
                              path=path,
                              data=data,
                              headers=headers,
                              query_params=query_params)

    def get(self,
            path: str,
            query_params: Optional[QueryParams] = None,
            headers: Headers = None,
            ) -> MLRPCResponse:
        return self._dispatch(method="GET",
                              path=path,
                              headers=headers,
                              query_params=query_params)

    def post(self,
             path: str,
             query_params: Optional[QueryParams] = None,
             data: Body = None,
             headers: Headers = None) -> MLRPCResponse:
        return self._dispatch(method="POST",
                              path=path,
                              headers=headers,
                              query_params=query_params,
                              data=data)

    def put(self, url,
            query_params: Optional[QueryParams] = None,
            data: Body = None,
            headers: Headers = None) -> MLRPCResponse:
        return self._dispatch(method="PUT",
                              path=url,
                              headers=headers,
                              query_params=query_params,
                              data=data)

    def patch(self, url,
              query_params: Optional[QueryParams] = None,
              data: Body = None,
              headers: Headers = None) -> MLRPCResponse:
        return self._dispatch(method="PATCH",
                              path=url,
                              headers=headers,
                              query_params=query_params,
                              data=data)

    def delete(self, url, headers: Headers = None) -> MLRPCResponse:
        return self._dispatch(method="DELETE",
                              path=url,
                              headers=headers)


class HotReloadMLRPCClient(MLRPCClient):

    def __init__(self, rpc_dispatch_handler: DispatchHandler):
        super().__init__(rpc_dispatch_handler)
        self._encrypt_decrypt = None

    def _setup_encryption(self):
        resp = self.get_public_key()[0]
        public_key = json.loads(resp.body)["public_key"]
        self._encrypt_decrypt = EncryptDecrypt(public_key=public_key)

    def hot_reload(self, directory_path) -> List[MLRPCResponse] | MLRPCResponse:
        if self._encrypt_decrypt is None:
            self._setup_encryption()
        from mlrpc.utils import dir_to_base64
        reload_dir = Path(directory_path)
        git_ignore = reload_dir / ".gitignore"
        content = dir_to_base64(reload_dir, git_ignore)
        return self._rpc_dispatch_handler.dispatch(HotReloadEvents.full_sync(content, self._encrypt_decrypt))

    def reinstall_requirements(self, requirements: List[str]) -> MLRPCResponse:
        return self._rpc_dispatch_handler.dispatch(HotReloadEvents.reinstall(requirements))

    def get_public_key(self) -> MLRPCResponse:
        return self._rpc_dispatch_handler.dispatch(HotReloadEvents.get_public_key())


class LocalServingDispatchHandler(DispatchHandler):

    def __init__(self, host: str = "0.0.0.0", port: int = 5000, endpoint: str = "/invocations"):
        self._host = host
        self._port = port
        self._endpoint = endpoint

    def dispatch(self, request: RequestObject) -> MLRPCResponse | List[MLRPCResponse]:
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx is required for LocalServingDispatchHandler; do pip install mlrpc[cli]")

        url = f"http://{self._host}:{self._port}{self._endpoint}"

        headers = request.headers or []
        headers.append(("Content-Type", "application/json"))
        # remove content-length header if present because the request is proxied and encoded in a custom way
        headers = [h for h in headers if h[0].lower() != "content-length"]
        content = request.encode().to_mlflow_df_split_dict()
        resp = httpx.request(method="POST", url=url, headers=headers,
                             json=content)
        predictions = resp.json()["predictions"]
        resp_objs = [ResponseObject.from_serving_resp(pred) for pred in predictions]
        return [MLRPCResponse(
            request=request,
            status_code=decoded_resp.status_code,
            headers=decoded_resp.headers,
            body=decoded_resp.content)
            for decoded_resp in resp_objs]


class ServingEndpointDispatchHandler(DispatchHandler):

    def __init__(self, endpoint_name: str, ws_client: WorkspaceClient = None):
        self._endpoint_name = endpoint_name
        self._ws = ws_client or WorkspaceClient()

    def dispatch(self, request: RequestObject) -> MLRPCResponse | List[MLRPCResponse]:
        serving_resp = self._ws.serving_endpoints.query(
            self._endpoint_name,
            dataframe_split=DataframeSplitInput(**request.encode().to_sdk_df_split())
        )
        decoded_resp = [ResponseObject.from_serving_resp(pred) for pred in serving_resp.predictions]
        return [MLRPCResponse(
            request=request,
            status_code=response.status_code,
            headers=response.headers,
            body=response.content
        )
            for response in decoded_resp]


class ServingRPCClient:

    def __init__(self, endpoint_name: str, ws_client: WorkspaceClient = None,
                 client_klass: Optional[Type[MLRPCClient] | Type[HotReloadMLRPCClient]] = None):
        client_klass = client_klass or MLRPCClient
        self._rpc_client = client_klass(ServingEndpointDispatchHandler(endpoint_name, ws_client))

    @property
    def rpc_client(self):
        return self._rpc_client


class LocalServingRPCClient:

    def __init__(self, host: str = "0.0.0.0", port: int = 6000, endpoint: str = "/invocations",
                 client_klass: Optional[Type[MLRPCClient] | Type[HotReloadMLRPCClient]] = None):
        client_klass = client_klass or MLRPCClient
        self._rpc_client = client_klass(LocalServingDispatchHandler(host, port, endpoint))

    @property
    def rpc_client(self):
        return self._rpc_client


class _Client:

    @staticmethod
    def local(host: str = "0.0.0.0", port: int = 6000, endpoint: str = "/invocations") -> MLRPCClient:
        return LocalServingRPCClient(host, port, endpoint).rpc_client

    @staticmethod
    def databricks(endpoint_name: str, ws_client: WorkspaceClient = None) -> MLRPCClient:
        return ServingRPCClient(endpoint_name, ws_client).rpc_client


class _HotReloadClient:

    @staticmethod
    def local(host: str = "0.0.0.0", port: int = 6000, endpoint: str = "/invocations") -> HotReloadMLRPCClient:
        return LocalServingRPCClient(host, port, endpoint, HotReloadMLRPCClient).rpc_client

    @staticmethod
    def databricks(endpoint_name: str, ws_client: WorkspaceClient = None) -> HotReloadMLRPCClient:
        return ServingRPCClient(endpoint_name, ws_client, HotReloadMLRPCClient).rpc_client


rpc = _Client()
hot_reload = _HotReloadClient()
