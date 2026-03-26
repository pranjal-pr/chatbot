import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


def wait_for_backend(health_url: str, timeout_sec: int, backend_process: subprocess.Popen) -> None:
    print(f"Waiting for backend health at {health_url} (timeout: {timeout_sec}s)", flush=True)
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if backend_process.poll() is not None:
            raise RuntimeError("Backend exited before becoming healthy.")

        try:
            with urllib.request.urlopen(health_url, timeout=1.5) as response:
                if response.status == 200:
                    print("Backend is healthy. Launching Streamlit.", flush=True)
                    return
        except (urllib.error.URLError, TimeoutError):
            pass

        time.sleep(0.75)

    raise RuntimeError(f"Backend health check timed out after {timeout_sec}s.")


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
    health_timeout_sec = int(os.getenv("API_HEALTH_TIMEOUT_SEC", "90"))

    child_env = os.environ.copy()
    child_env.setdefault("API_URL", f"http://{api_host}:{api_port}")

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

    backend_process = subprocess.Popen(backend_cmd, env=child_env)
    frontend_process: subprocess.Popen | None = None

    def handle_shutdown(signum, _frame) -> None:
        terminate_process(frontend_process)
        terminate_process(backend_process)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    try:
        wait_for_backend(
            health_url=f"http://{api_host}:{api_port}/health",
            timeout_sec=health_timeout_sec,
            backend_process=backend_process,
        )
        frontend_process = subprocess.Popen(streamlit_cmd, env=child_env)
        return frontend_process.wait()
    finally:
        terminate_process(frontend_process)
        terminate_process(backend_process)


if __name__ == "__main__":
    raise SystemExit(main())
