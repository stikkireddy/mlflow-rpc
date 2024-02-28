import abc
import json
from dataclasses import dataclass
from typing import Sequence, Tuple, Literal, Dict, Union, List, Optional, Any
from urllib.parse import urlencode

from mlrpc.flavor import RequestObject, ResponseObject

MethodType = Literal['GET', 'POST', 'PUT', 'DELETE', 'PATCH']
QueryParams = Dict[str, Union[str, List[str]]]
Headers = Sequence[Tuple[str, str]]
Body = Optional[str | Dict[str, Any] | List[Dict[str, Any]]]


@dataclass
class MLRPCResponse:
    request: RequestObject
    status_code: int
    headers: Optional[Sequence[Tuple[str, str]]] = None
    body: Optional[str] = None

    @property
    def dict(self):
        return json.loads(self.body)

    @property
    def json(self):
        return self.dict

    @property
    def text(self):
        return str(self.body)


def generate_query_string(params: QueryParams) -> str:
    return urlencode(params, doseq=True)


class DispatchHandler(abc.ABC):

    @abc.abstractmethod
    def dispatch(self, request: RequestObject) -> MLRPCResponse | List[MLRPCResponse]:
        pass


class MLRPCClient:
    def __init__(self, rpc_dispatch_handler: DispatchHandler):
        self._rpc_dispatch_handler = rpc_dispatch_handler

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


class MLFlowURIDispatchHandler(DispatchHandler):

    def __init__(self, uri: str):
        self._uri = uri

    def dispatch(self, request: RequestObject) -> MLRPCResponse | List[MLRPCResponse]:
        import mlflow
        m = mlflow.pyfunc.load_model(self._uri)
        resp = m.predict(request.encode().to_df())
        responses = ResponseObject.from_mlflow_predict(resp)
        return [MLRPCResponse(
            request=request,
            status_code=200,
            headers=response.headers,
            body=response.content
        )
            for response in responses]


class MLFlowRPCClient(MLRPCClient):

    def __init__(self, uri: str):
        super().__init__(MLFlowURIDispatchHandler(uri))
