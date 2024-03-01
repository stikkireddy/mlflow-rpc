import os

from fastapi import FastAPI, Request
from fastapi.openapi.docs import get_swagger_ui_html
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response, JSONResponse


def make_swagger_proxy(
        endpoint_name: str,
        profile: str = "default",
        port: int = 8000,
        debug: bool = False,
        databricks_mode: bool = True,
        local_server_port: int = 6500
):
    class RawRequestMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            # Load the raw request
            raw_request = await request.body()

            from mlrpc.client import rpc

            def get_just_path(request: Request):
                return str(request.url).replace(str(request.base_url), "/")

            path = get_just_path(request)
            if path.startswith("/docs"):
                return get_swagger_ui_html(
                    openapi_url=f"http://0.0.0.0:{port}/openapi.json",
                    title="MLRPC Swagger UI",
                )

            if databricks_mode is True:
                os.environ["DATABRICKS_CONFIG_PROFILE"] = profile
                rpc_flavor = rpc.databricks(endpoint_name)
                mlrpc_response = rpc_flavor \
                    .request(str(request.method), path, headers=request.headers, data=raw_request.decode("utf-8"))
            else:
                rpc_flavor = rpc.local(port=local_server_port)
                mlrpc_response = rpc_flavor \
                    .request(str(request.method), path, headers=[(k, v) for k, v in request.headers.items()],
                             data=raw_request.decode("utf-8"))


            headers = {k: v for d in mlrpc_response.headers for k, v in d.items() if k.lower() != "content-length"}
            if debug is True:
                print("MLRPC RESPONSE: ", mlrpc_response)
            if isinstance(mlrpc_response.body, str):
                content = mlrpc_response.body
                response = Response(content=content, status_code=mlrpc_response.status_code,
                                    headers=headers)
            else:
                response = JSONResponse(content=mlrpc_response.body, status_code=mlrpc_response.status_code,
                                        headers=headers)

            return response

    app = FastAPI()

    @app.middleware("http")
    async def add_raw_request_middleware(request: Request, call_next):
        middleware = RawRequestMiddleware(app)
        return await middleware.dispatch(request, call_next)

    return app
