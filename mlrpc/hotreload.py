import queue
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal, List
from typing import Optional

from databricks.sdk import WorkspaceClient
from dateutil.parser import parse
from pathspec import PathSpec
from pathspec.patterns import GitWildMatchPattern
from pytz import timezone
from watchdog.events import FileSystemEventHandler, FileSystemEvent
from watchdog.observers import Observer

from mlrpc.client import MLRPCResponse, HotReloadMLRPCClient
from mlrpc.utils import get_requirements_from_file


def get_gitignore_specs(dir_to_watch) -> Optional[PathSpec]:
    ignore_file = Path(dir_to_watch) / ".gitignore"
    if ignore_file is not None and ignore_file.exists() and ignore_file.is_file():
        ignore_file = str(ignore_file)
        with open(ignore_file, 'r') as f:
            gitignore = f.read()
        return PathSpec.from_lines(GitWildMatchPattern, gitignore.splitlines())

    return None


def maybe_requirements_txt_change(src_path: str, rpc_client: HotReloadMLRPCClient) -> Optional[MLRPCResponse]:
    if src_path.rstrip("~").endswith("requirements.txt"):
        requirements = get_requirements_from_file(Path(src_path))
        if requirements:
            response = rpc_client.reinstall_requirements(requirements)
            return response


def hot_reload_on_change(dir_to_watch, rpc_client: HotReloadMLRPCClient, frequency_seconds: int = 1,
                         logging_function: Callable = None,
                         error_logging_function: Callable = None,
                         success_logging_function: Callable = None):
    logging_function = logging_function or print
    error_logging_function = error_logging_function or print
    event_queue: queue.Queue[FileSystemEvent] = queue.Queue()

    logging_function("Attempting to start hot reload...")

    class FileChangeHandler(FileSystemEventHandler):
        def on_any_event(self, event: FileSystemEvent):
            event_queue.put(event)

    def consumer():
        logging_function(f"Starting file watcher for {dir_to_watch}...")

        def handle_response(_response: MLRPCResponse | List[MLRPCResponse],
                            event_type: Literal["hot-reload", "reinstall-pip-requirements"]):
            if response.status_code != 200:
                error_logging_function(
                    f"Event: {event_type} failed with status: {_response.status_code} - {_response.body}")
                return

            success_logging_function(f"Event: {event_type} status: {_response.status_code} - {_response.body}")

        while True:
            # Collect all changes made in the last 5 seconds
            any_changes = []
            start_time = time.time()
            while time.time() - start_time < frequency_seconds:
                try:
                    event = event_queue.get(timeout=1)
                    any_changes.append(event)
                except queue.Empty:
                    pass

            # If there are any changes, call full_sync
            if any_changes:
                ignore_specs = get_gitignore_specs(dir_to_watch)
                valid_changes = []
                for change in any_changes:
                    # if ignore specs is there and doesnt match its valid file
                    if ignore_specs is not None and not ignore_specs.match_file(change.src_path) \
                            and change.is_directory is False and change.src_path.endswith("~") is False:
                        valid_changes.append(change)
                    # if ignore specs is not there, all changes are valid
                    if ignore_specs is None:
                        valid_changes.append(change)
                logging_function(f"Found changes: {len(any_changes)} - Valid changes: {len(valid_changes)}")
                if not valid_changes:
                    continue

                for change in valid_changes:
                    responses = maybe_requirements_txt_change(change.src_path, rpc_client)
                    if responses is not None:
                        if isinstance(responses, list):
                            for response in responses:
                                handle_response(response, "reinstall-pip-requirements")
                        else:
                            handle_response(responses, "reinstall-pip-requirements")

                logging_function(f"Files changed firing hot reload for {dir_to_watch}...")
                responses = rpc_client.hot_reload(dir_to_watch)

                if responses is not None:
                    if isinstance(responses, list):
                        for response in responses:
                            handle_response(response, "hot-reload")
                    else:
                        handle_response(responses, "hot-reload")

    consumer_thread = threading.Thread(target=consumer, daemon=True)
    consumer_thread.start()

    event_handler = FileChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, dir_to_watch, recursive=True)
    observer.start()

    return [consumer_thread, observer]





def extract_second_box(log_line):
    matches = re.findall(r'\[.*?\]', log_line)
    if len(matches) >= 2:
        return matches[1][1:-1]  # remove brackets
    else:
        return None


def extract_iso_timestamp(possible_ts: Optional[str]):
    if possible_ts is None:
        return None
    try:
        naive_dt = parse(possible_ts, ignoretz=True)
        utc_dt = timezone('UTC').localize(naive_dt).replace(tzinfo=None)
        return utc_dt
    except ValueError:
        return None


class LogMonitor:
    def __init__(self, ws: WorkspaceClient,
                 endpoint_name: str,
                 from_beginning: bool = False,
                 logging_function: Optional[callable] = None,
                 ):
        self.ws_client = ws
        self.endpoint_name = endpoint_name
        self.model_names = []
        for se in ws.serving_endpoints.get(endpoint_name).config.served_entities:
            self.model_names.append(se.name)
        now = datetime.utcnow().replace(tzinfo=None)
        self._last_log_ts = {mn: None for mn in self.model_names} if from_beginning else {mn: now for mn in
                                                                                          self.model_names}
        if from_beginning is False:
            logging_function("Looking for logs from", now)
        self._logging_function = logging_function or print

    def print_logs_if_havent_been_seen(self):
        logs = []
        for model_name in self.model_names:
            resp = self.ws_client.serving_endpoints.logs(self.endpoint_name, model_name)
            # read logs backwards and stop when we see a timestamp we've seen before
            latest_ts = None
            for idx, line in enumerate(reversed(resp.logs.splitlines())):
                ts = extract_iso_timestamp(extract_second_box(line))
                if ts is None:
                    continue
                if latest_ts is None:
                    latest_ts = ts
                if self._last_log_ts[model_name] is not None and ts <= self._last_log_ts[model_name]:
                    break
                logs.append(line)
            self._last_log_ts[model_name] = latest_ts

            for line in reversed(logs):
                self._logging_function(line)
        return logs


def make_log_monitor_thread(ws: WorkspaceClient,
                            endpoint_name: str,
                            from_beginning: bool = False,
                            logging_function: Optional[callable] = None):
    def _monitor():
        monitor = LogMonitor(ws, endpoint_name, from_beginning, logging_function=logging_function)
        while True:
            try:
                monitor.print_logs_if_havent_been_seen()
            except Exception as e:
                logging_function(f"Error in log monitor: {e}")
            finally:
                time.sleep(10)

    return threading.Thread(target=_monitor, daemon=True)
