"""Start the local Home-Service Agent stack in dependency order."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Sequence
from urllib.error import URLError
from urllib.request import urlopen

from dotenv import load_dotenv

from home_service_agent.storage import StoragePaths, initialize_storage


ROOT = Path(__file__).resolve().parent
UI_ROOT = ROOT / "ui" / "home-service-agent-ui"


def build_commands(npm: str) -> dict[str, list[str]]:
    """Return the three local processes without embedding any secrets."""

    return {
        "langgraph": [npm, "run", "dev", "--workspace", "langgraph-server"],
        "api": [sys.executable, "-m", "uvicorn", "home_service_agent.app:app", "--host", "127.0.0.1", "--port", "8000"],
        "ui": [npm, "run", "dev", "--workspace", "web"],
    }


def prepare_environment(paths: StoragePaths) -> dict[str, str]:
    """Build child-process environment after loading local, ignored settings."""

    environment = os.environ.copy()
    environment["HOME_SERVICE_AGENT_CHECKPOINT_DB"] = str(paths.checkpoints_db)
    environment["NEXT_PUBLIC_API_URL"] = "http://127.0.0.1:2024"
    environment["NEXT_PUBLIC_ASSISTANT_ID"] = "home_service_agent"
    python313 = shutil.which("python3.13")
    if python313:
        # The Node CLI launches Python in an isolated environment; Python 3.13
        # avoids the native jsonschema-rs compatibility issue on Python 3.14.
        environment["UV_PYTHON"] = python313
    return environment


def wait_for_http(url: str, process: subprocess.Popen[str], *, timeout: float = 45.0) -> None:
    """Wait for a local readiness endpoint, failing if its process exits."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"process exited before becoming ready: {url}")
        try:
            with urlopen(url, timeout=1) as response:
                if 200 <= response.status < 300:
                    return
        except (OSError, URLError):
            time.sleep(0.25)
    raise TimeoutError(f"timed out waiting for {url}")


def terminate_processes(processes: Sequence[subprocess.Popen[str]]) -> None:
    """Stop child process groups so npm's descendants are not orphaned."""

    for process in reversed(processes):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 8
    for process in processes:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def start_stack() -> int:
    """Bootstrap storage, start services, and keep them alive until interrupted."""

    load_dotenv(ROOT / ".env")
    data_dir = os.getenv("HOME_SERVICE_AGENT_DATA_DIR", str(ROOT / "data"))
    paths = initialize_storage(data_dir)
    npm = shutil.which("npm")
    if npm is None:
        raise RuntimeError("npm is required to start the local Agent Chat UI")

    environment = prepare_environment(paths)
    commands = build_commands(npm)
    processes: list[subprocess.Popen[str]] = []
    try:
        langgraph_environment = environment.copy()
        # The CLI runs the graph in a separate dependency environment. Keep
        # its nested ChatAgent in memory; the SQLite checkpointer is used by
        # the project runtime and FastAPI process, where versions are aligned.
        langgraph_environment.pop("HOME_SERVICE_AGENT_CHECKPOINT_DB", None)
        print(f"Initialized local storage in {paths.data_dir}")
        print("Starting LangGraph server on http://127.0.0.1:2024")
        processes.append(
            subprocess.Popen(
                commands["langgraph"],
                cwd=UI_ROOT,
                env=langgraph_environment,
                start_new_session=True,
            )
        )
        wait_for_http("http://127.0.0.1:2024/info", processes[-1])

        print("Starting FastAPI compatibility service on http://127.0.0.1:8000")
        processes.append(
            subprocess.Popen(commands["api"], cwd=ROOT, env=environment, start_new_session=True)
        )
        wait_for_http("http://127.0.0.1:8000/health", processes[-1])

        print("Starting Agent Chat UI on http://127.0.0.1:3000")
        processes.append(
            subprocess.Popen(commands["ui"], cwd=UI_ROOT, env=environment, start_new_session=True)
        )
        processes[-1].wait()
        return processes[-1].returncode or 0
    finally:
        terminate_processes(processes)


def main() -> int:
    try:
        return start_stack()
    except KeyboardInterrupt:
        return 130
    except (OSError, RuntimeError, TimeoutError) as error:
        print(f"Startup failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
