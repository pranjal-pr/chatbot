import pytest

import start


def test_backend_client_host_rewrites_wildcard_ipv4():
    assert start.backend_client_host("0.0.0.0") == "127.0.0.1"


def test_backend_client_host_rewrites_wildcard_ipv6():
    assert start.backend_client_host("::") == "127.0.0.1"
    assert start.backend_client_host("[::]") == "127.0.0.1"


def test_backend_client_host_keeps_connectable_host():
    assert start.backend_client_host("127.0.0.1") == "127.0.0.1"
    assert start.backend_client_host("backend") == "backend"


def test_main_stops_backend_when_frontend_fails_to_start(monkeypatch):
    class FakeBackendProcess:
        terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout):
            return 0

    backend_process = FakeBackendProcess()
    popen_calls = 0

    def fake_popen(*_args, **_kwargs):
        nonlocal popen_calls
        popen_calls += 1
        if popen_calls == 1:
            return backend_process
        raise OSError("Streamlit failed to start")

    monkeypatch.setattr(start.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(start.signal, "signal", lambda *_args: None)

    with pytest.raises(OSError, match="Streamlit failed to start"):
        start.main()

    assert backend_process.terminated is True
