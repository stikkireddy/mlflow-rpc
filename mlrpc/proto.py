import base64
import hashlib
import io
import json
import os
import shutil
import tarfile
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Dict, Literal, List


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
    def from_httpx_resp(cls, resp: "httpx.Response"):
        # TODO: being lazy here, should handle more gracefully using mimetype
        try:
            return cls(
                status_code=resp.status_code,
                headers=list(resp.headers.items()),
                content=resp.json()
            )
        except Exception:
            return cls(
                status_code=resp.status_code,
                headers=list(resp.headers.items()),
                content=resp.text
            )


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


# HOT RELOAD EVENTS ARE ALWAYS POSTS
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
    def reinstall(requirements: List[str]) -> RequestObject:
        return RequestObject(
            method="POST",
            path="/__INTERNAL__/REINSTALL",
            content=json.dumps({
                "requirements": requirements
            })
        )
