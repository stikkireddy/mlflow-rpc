import base64
import hashlib
import io
import json
import os
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Dict, Literal, List, Callable

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


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
    base64_data = base64.b64decode(base64_string)

    data = io.BytesIO(base64_data)

    shutil.rmtree(target_dir)
    os.makedirs(target_dir)

    with tarfile.open(fileobj=data, mode='r:gz') as tar:
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
    def full_sync(content: str, encryption_key: "EncryptDecrypt") -> RequestObject:
        enc_content = encryption_key.encrypt(content)
        return RequestObject(
            method="POST",
            path="/__INTERNAL__/FULL_SYNC",
            content=json.dumps({
                "content": enc_content,
                "checksum": hashlib.md5(content.encode('utf-8')).hexdigest()
            })
        )

    @staticmethod
    def upload_large_file(c: "Chunk"):
        return RequestObject(
            method="POST",
            path="/__INTERNAL__/UPLOAD_LARGE_FILE",
            content=json.dumps({
                "chunk": c.dict()
            })
        )


    @staticmethod
    def reload() -> RequestObject:
        return RequestObject(
            method="POST",
            path="/__INTERNAL__/RELOAD"
        )

    @staticmethod
    def get_public_key() -> RequestObject:
        return RequestObject(
            method="POST",
            path="/__INTERNAL__/GET_PUBLIC_KEY"
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


class KeyGenerator:
    def __init__(self, directory, debug_msg: Optional[Callable] = None):
        self.directory = directory
        self.private_key = None
        self.public_key = None
        self._debug_msg = debug_msg or print
        self._public_key_name = 'public_key.pem'
        self._private_key_name = 'private_key.pem'

    def generate(self):
        private_key_path = os.path.join(self.directory, self._private_key_name)
        public_key_path = os.path.join(self.directory, self._public_key_name)

        if os.path.exists(private_key_path) and os.path.exists(public_key_path):
            self._debug_msg(f"Keys already exist in {self.directory}")
            return

        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        self.public_key = self.private_key.public_key()

        private_key_pem = self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        public_key_pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        with open(private_key_path, 'wb') as f:
            f.write(private_key_pem)

        with open(public_key_path, 'wb') as f:
            f.write(public_key_pem)

        self._debug_msg(f"Keys generated and saved to {self.directory}")

    def _get_or_regenerate(self, key_name):
        if os.path.exists(os.path.join(self.directory, key_name)) is False:
            self._debug_msg(f"{key_name} does not exist in {self.directory} so regenerating...")
            self.generate()
        with open(os.path.join(self.directory, key_name), 'rb') as f:
            return f.read().decode()

    def get_public_key(self):
        return self._get_or_regenerate(self._public_key_name)

    def get_private_key(self):
        return self._get_or_regenerate(self._private_key_name)


class EncryptDecrypt:
    def __init__(self, *,
                 private_key=None,
                 public_key=None,
                 key_generator: Optional[KeyGenerator] = None):
        self.private_key = serialization.load_pem_private_key(
            private_key.encode(),
            password=None
        ) if private_key else None
        self.public_key = serialization.load_pem_public_key(
            public_key.encode()
        ) if public_key else None
        self.key_generator = key_generator
        self.chunk_delimiter = '  '

    def encrypt(self, message: str) -> str:
        if not self.public_key and self.key_generator is not None:
            public_key = self.key_generator.get_public_key()
            self.public_key = serialization.load_pem_public_key(
                public_key.encode()
            )
        if not self.public_key:
            raise ValueError("Public key is not set")

        # magic number for 2048 bit RSA key
        chunk_size = 150

        # encrypt chunks
        encrypted_chunks = []
        for i in range(0, len(message), chunk_size):
            chunk = message[i:i + chunk_size]
            encrypted_chunk = self.public_key.encrypt(
                chunk.encode(),
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            encrypted_chunks.append(base64.b64encode(encrypted_chunk).decode())

        encrypted_content = self.chunk_delimiter.join(encrypted_chunks)
        return encrypted_content

    def decrypt(self, b64_ciphertext: str):
        if not self.private_key and self.key_generator is not None:
            private_key = self.key_generator.get_private_key()
            self.private_key = serialization.load_pem_private_key(
                private_key.encode(),
                password=None
            )

        if not self.private_key:
            raise ValueError("Private key is not set")

        encrypted_chunks = b64_ciphertext.split(self.chunk_delimiter)
        decrypted_chunks = []

        for chunk in encrypted_chunks:
            ciphertext = base64.b64decode(chunk)
            plaintext = self.private_key.decrypt(
                ciphertext,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            decrypted_chunks.append(plaintext.decode())

        return ''.join(decrypted_chunks)


@dataclass
class Chunk:
    content_b64: str
    chunk_number: int
    next_offset: int
    encrypted: bool = False
    eof: bool = False
    start_of_file: bool = False
    permissions: Optional[str] = None
    relative_file_path: Optional[str] = None
    # TODO: include checksums

    def dict(self):
        return {
            "content_b64": self.content_b64,
            "chunk_number": self.chunk_number,
            "next_offset": self.next_offset,
            "encrypted": self.encrypted,
            "eof": self.eof,
            "start_of_file": self.start_of_file,
            "permissions": self.permissions,
            "relative_file_path": self.relative_file_path
        }

    def __str__(self):
        return (f"Chunk {self.chunk_number} - {self.next_offset} - {self.encrypted} - "
                f"{self.eof} - {self.start_of_file} - {self.permissions} - {self.relative_file_path}")

    def __repr__(self):
        return self.__str__()


class DataFileChunker:

    def __init__(self,
                 root_dir: str,
                 data_dir: str,
                 chunk_size_bytes: int = 10000000,
                 encrypt_decrypt: Optional[EncryptDecrypt] = None):
        self.data_dir = data_dir
        self.root_dir = root_dir
        self.chunk_size_bytes = chunk_size_bytes
        self.encrypt_decrypt = encrypt_decrypt

    def _get_offset(self, last_chunk: Optional[Chunk] = None) -> int:
        if last_chunk is None:
            return 0

        return last_chunk.next_offset or 0

    def _is_eof(self, last_chunk: Optional[Chunk] = None) -> bool:
        if last_chunk is None:
            return False
        return last_chunk.eof

    def make_chunk(self, file_path: str, last_chunk: Optional[Chunk] = None) -> Chunk:
        with open(file_path, 'rb') as f:
            offset = self._get_offset(last_chunk)
            f.seek(offset)
            content = f.read(self.chunk_size_bytes)
            content_b64 = base64.b64encode(content).decode()
            if self.encrypt_decrypt:
                content_b64 = self.encrypt_decrypt.encrypt(content_b64)

            return Chunk(
                relative_file_path=Path(file_path).relative_to(self.root_dir).as_posix(),
                start_of_file=offset == 0,
                permissions=oct(os.stat(file_path).st_mode & 0o777),
                content_b64=content_b64,
                chunk_number=last_chunk.chunk_number + 1 if last_chunk else 0,
                next_offset=f.tell(),
                encrypted=bool(self.encrypt_decrypt),
                eof=f.tell() == os.fstat(f.fileno()).st_size
            )

    def make_n_chunks(self, file_path: str, n: int, last_chunk: Optional[Chunk] = None) -> List[Chunk]:
        chunks = []
        for _ in range(n):
            if self._is_eof(last_chunk):
                break
            chunk = self.make_chunk(file_path, last_chunk=last_chunk)
            chunks.append(chunk)
            last_chunk = chunk
        return chunks

    def iter_chunks(self, file_path: str, last_chunk: Optional[Chunk] = None):
        while True:
            if self._is_eof(last_chunk):
                break
            chunk = self.make_chunk(file_path, last_chunk=last_chunk)
            yield chunk
            last_chunk = chunk

    def iter_files_chunks(self):
        root_path = Path(self.root_dir)
        data_path = root_path / self.data_dir
        for file in data_path.rglob("*"):
            for chunk in self.iter_chunks(str(file)):
                yield chunk

    def iter_requests(self):
        for chunk in self.iter_files_chunks():
            yield HotReloadEvents.upload_large_file(c=chunk)

class DataFileChunkWriter:

    def __init__(self,
                 root_dir: str,
                 encrypt_decrypt: Optional[EncryptDecrypt] = None):
        self.root_dir = root_dir
        self.encrypt_decrypt = encrypt_decrypt

    def write_chunk(self, chunk: Chunk):
        target_file = Path(self.root_dir) / chunk.relative_file_path
        target_file.parent.mkdir(parents=True, exist_ok=True)
        write_mode = 'wb' if chunk.start_of_file else 'ab'
        with target_file.open(write_mode) as f:
            content = base64.b64decode(chunk.content_b64)
            if chunk.encrypted:
                decrypted = self.encrypt_decrypt.decrypt(chunk.content_b64)
                content: bytes = base64.b64decode(decrypted)
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        if chunk.permissions is not None:
            os.chmod(target_file, int(chunk.permissions, 8))
        return True

    def write_chunks(self, chunks: List[Chunk]):
        for chunk in chunks:
            self.write_chunk(chunk)
        return True
