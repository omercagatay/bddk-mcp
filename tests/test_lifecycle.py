"""Integration tests for server lifecycle."""

import time
from deps import Dependencies


def test_deps_creation():
    """Dependencies can be created with None values for unit tests."""
    deps = Dependencies(pool=None, doc_store=None, client=None, http=None)
    assert deps.server_start_time <= time.time()
    assert deps.vector_store is None
    assert deps.sync_circuit_open is False


def test_deps_health_state_tracking():
    """Dependencies tracks health state correctly."""
    deps = Dependencies(pool=None, doc_store=None, client=None, http=None)

    # Simulate failures
    deps.sync_consecutive_failures = 10
    deps.sync_circuit_open = True
    deps.last_sync_error = "Connection refused"

    assert deps.sync_circuit_open is True
    assert deps.last_sync_error == "Connection refused"

    # Reset
    deps.sync_consecutive_failures = 0
    deps.sync_circuit_open = False
    deps.last_sync_time = time.time()
    deps.last_sync_error = None

    assert deps.sync_circuit_open is False
    assert deps.last_sync_time is not None


def test_all_tool_modules_importable():
    """All tool modules can be imported and have register()."""
    from tools import admin, analytics, bulletin, documents, search, sync
    for mod in [admin, analytics, bulletin, documents, search, sync]:
        assert hasattr(mod, "register"), f"{mod.__name__} missing register()"
        assert callable(mod.register)


def test_sync_module_has_startup_sync():
    """sync module exposes startup_sync for server.py."""
    from tools.sync import startup_sync
    import asyncio
    assert asyncio.iscoroutinefunction(startup_sync)
