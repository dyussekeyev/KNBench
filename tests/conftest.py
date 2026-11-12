"""Shared pytest configuration.

Every test must run offline. ``respx`` mocks the OpenRouter transport, and this
guard makes an unmocked request fail loudly instead of reaching the internet.
Loopback names stay allowed because the asyncio event loop uses them on Windows.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator

import pytest

_ALLOWED_HOSTS = {"localhost", "127.0.0.1", "::1", None}


@pytest.fixture(autouse=True)
def block_external_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    real_getaddrinfo = socket.getaddrinfo

    def guarded(host: str | None, *args: object, **kwargs: object) -> object:
        if host not in _ALLOWED_HOSTS:
            raise RuntimeError(f"external network access attempted: {host}")
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", guarded)
    yield
