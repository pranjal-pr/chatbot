import os
import signal
import subprocess
import sys
import time

from dotenv import load_dotenv

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(PROJECT_DIR, ".env"))


def backend_client_host(bind_host: str) -> str:
    """Return a host that clients can connect to when uvicorn binds to all interfaces."""
    normalized_host = bind_host.strip().lower()
    if normalized_host in {"0.0.0.0", "::", "[::]"}:
        return "127.0.0.1"
    return bind_host


def terminate_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def main() -> int:
    api_host = os.getenv("API_HOST", "127.0.0.1")
    api_port = os.getenv("API_PORT", "8000")
    app_host = os.getenv("APP_HOST", "0.0.0.0")
    app_port = os.getenv("PORT", "7860")

    child_env = os.environ.copy()
    child_env.setdefault("API_URL", f"http://{backend_client_host(api_host)}:{api_port}")
    child_env.setdefault("PYTHONUNBUFFERED", "1")

    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "api:app",
        "--host",
        api_host,
        "--port",
        api_port,
    ]
    streamlit_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "chatbot.py",
        "--server.enableCORS",
        "false",
        "--server.enableXsrfProtection",
        "false",
        "--server.address",
        app_host,
        "--server.port",
        app_port,
        "--server.headless",
        "true",
    ]

    backend_process: subprocess.Popen | None = None
    frontend_process: subprocess.Popen | None = None

    def handle_shutdown(signum, _frame) -> None:
        terminate_process(frontend_process)
        terminate_process(backend_process)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    try:
        print(f"Starting backend on http://{api_host}:{api_port}", flush=True)
        backend_process = subprocess.Popen(backend_cmd, env=child_env, cwd=PROJECT_DIR)
        print(f"Starting Streamlit on http://{app_host}:{app_port}", flush=True)
        frontend_process = subprocess.Popen(streamlit_cmd, env=child_env, cwd=PROJECT_DIR)

        while True:
            frontend_exit_code = frontend_process.poll()
            if frontend_exit_code is not None:
                return frontend_exit_code

            backend_exit_code = backend_process.poll()
            if backend_exit_code is not None:
                print(
                    f"Backend exited with status {backend_exit_code}; stopping Streamlit.", file=sys.stderr, flush=True
                )
                return backend_exit_code or 1

            time.sleep(1)
    finally:
        terminate_process(frontend_process)
        terminate_process(backend_process)


if __name__ == "__main__":
    raise SystemExit(main())
