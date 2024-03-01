import socket
from dataclasses import dataclass
from typing import List


def ensure_python_path(env):
    import sys
    from pathlib import Path
    for python_version_dir in (Path(sys.executable).parent.parent / "lib").iterdir():
        site_packages = str(python_version_dir / "site-packages")
        py_path = env.get("PYTHONPATH", "")
        if site_packages not in py_path.split(":"):
            env["PYTHONPATH"] = f"{py_path}:{site_packages}"


def execute(*, cmd: List[str], env, cwd=None, ensure_python_site_packages=True, shell=False, trim_new_line=True):
    if ensure_python_site_packages:
        ensure_python_path(env)
    import subprocess
    if shell is True:
        cmd = " ".join(cmd)
    popen = subprocess.Popen(cmd,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT,
                             universal_newlines=True,
                             shell=shell,
                             env=env,
                             cwd=cwd,
                             bufsize=1)
    if popen.stdout is not None:
        for stdout_line in iter(popen.stdout.readline, ""):
            if trim_new_line:
                stdout_line = stdout_line.strip()
            yield stdout_line

    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def find_next_open_port(start_port: int = 6500, end_port: int = 7000) -> int:
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise Exception(f"No open ports found in the range {start_port}-{end_port}")


@dataclass
class DatabricksProfile:
    name: str
    host: str
    token: str


def get_profile_contents(profile_name, profile_path: str = "~/.databrickscfg") -> DatabricksProfile:
    from configparser import ConfigParser
    import os
    profile_path = os.path.expanduser(profile_path)
    config = ConfigParser()
    config.read(profile_path)
    host = config[profile_name]["host"]
    token = config[profile_name]["token"]
    return DatabricksProfile(name=profile_name, host=host, token=token)


def get_version(package_name: str = "mlrpc") -> str:
    try:
        from importlib import metadata  # type: ignore
    except ImportError:
        # Python < 3.8
        import importlib_metadata as metadata  # type: ignore

    try:
        return metadata.version(package_name)  # type: ignore
    except metadata.PackageNotFoundError:  # type: ignore
        return "unknown"
