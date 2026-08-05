"""
Unit tests for the workhorse hub-offline fallback: endpoint parsing, the
TCP probe (socket stubbed both ways), and the env-setting decision logic.
No network, no downloads.
"""

import os
import socket

import pytest

from pytti import workhorse

OFFLINE_KEYS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")


@pytest.fixture
def clean_hub_env(monkeypatch):
    """No offline env vars during the test; whatever the test set is removed
    afterwards (monkeypatch then restores any genuine pre-existing values)."""
    for key in OFFLINE_KEYS:
        monkeypatch.delenv(key, raising=False)
    yield
    for key in OFFLINE_KEYS:
        os.environ.pop(key, None)


def _probe_must_not_run() -> bool:
    raise AssertionError("probe ran despite a pre-set offline env var")


# ----------------------------------------------------------------------
# endpoint parsing
# ----------------------------------------------------------------------


def test_hub_host_port_default(monkeypatch):
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    assert workhorse._hub_host_port() == ("huggingface.co", 443)


def test_hub_host_port_empty_endpoint_falls_back(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "   ")
    assert workhorse._hub_host_port() == ("huggingface.co", 443)


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [
        ("https://hub-mirror.example", ("hub-mirror.example", 443)),
        ("http://hub-mirror.example", ("hub-mirror.example", 80)),
        ("http://localhost:8080", ("localhost", 8080)),
    ],
)
def test_hub_host_port_honors_hf_endpoint(monkeypatch, endpoint, expected):
    monkeypatch.setenv("HF_ENDPOINT", endpoint)
    assert workhorse._hub_host_port() == expected


def test_hub_host_port_hostless_endpoint_fails_loud(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "not-a-url")
    with pytest.raises(ValueError, match="no hostname"):
        workhorse._hub_host_port()


# ----------------------------------------------------------------------
# probe (socket stubbed both ways)
# ----------------------------------------------------------------------


class _FakeConnection:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_hub_reachable_when_connect_succeeds(monkeypatch):
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    connection = _FakeConnection()
    calls = []

    def fake_create_connection(address, timeout):
        calls.append((address, timeout))
        return connection

    monkeypatch.setattr(socket, "create_connection", fake_create_connection)
    assert workhorse.hub_reachable() is True
    assert calls == [(("huggingface.co", 443), workhorse.HUB_PROBE_TIMEOUT_S)]
    assert connection.closed  # no leaked socket


@pytest.mark.parametrize("error", [socket.gaierror("dns down"), TimeoutError(), OSError("no route")])
def test_hub_unreachable_on_socket_errors(monkeypatch, error):
    def fake_create_connection(address, timeout):
        raise error

    monkeypatch.setattr(socket, "create_connection", fake_create_connection)
    assert workhorse.hub_reachable() is False


# ----------------------------------------------------------------------
# fallback decision
# ----------------------------------------------------------------------


def test_unreachable_hub_engages_offline_env(clean_hub_env):
    assert workhorse.configure_offline_fallback(probe=lambda: False) is True
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_reachable_hub_leaves_env_unset(clean_hub_env):
    assert workhorse.configure_offline_fallback(probe=lambda: True) is False
    assert "HF_HUB_OFFLINE" not in os.environ
    assert "TRANSFORMERS_OFFLINE" not in os.environ


@pytest.mark.parametrize("key", OFFLINE_KEYS)
@pytest.mark.parametrize("value", ["1", "0"])
def test_preset_offline_env_skips_probe_and_is_never_overridden(
    clean_hub_env, monkeypatch, key, value
):
    monkeypatch.setenv(key, value)
    assert workhorse.configure_offline_fallback(probe=_probe_must_not_run) is False
    assert os.environ[key] == value
    other = next(k for k in OFFLINE_KEYS if k != key)
    assert other not in os.environ
