"""The libsumo fallback warning names a fix that works outside pixi too.

``libsumo`` is its own PyPI package, versioned with SUMO itself, so a pip
user whose install lacks it needs ``pip install libsumo==<SUMO version>``;
the warning used to tell everyone to run ``pixi install``.
"""
import importlib
import sys
import warnings

from opencda_marl._kernel.traffic import _sumo_home, _sumo_lib


def _reload_without_libsumo(monkeypatch, resolved):
    # Restored on teardown: the module's binding and the resolution cache.
    monkeypatch.setattr(_sumo_lib, "traci", _sumo_lib.traci)
    monkeypatch.setattr(_sumo_lib, "SUMO_BINDING", _sumo_lib.SUMO_BINDING)
    monkeypatch.setattr(_sumo_lib, "_SUMO_INFO", _sumo_lib._SUMO_INFO)
    monkeypatch.setattr(_sumo_home, "_RESOLVED", resolved)
    monkeypatch.setenv("OPENCDA_SUMO_LIB", "libsumo")
    monkeypatch.setitem(sys.modules, "libsumo", None)   # import raises ImportError
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(_sumo_lib)
    messages = [str(w.message) for w in caught
                if issubclass(w.category, RuntimeWarning) and "libsumo" in str(w.message)]
    assert len(messages) == 1, messages
    return messages[0]


def _resolution(binary_version):
    return {"home": "sumo", "source": "eclipse-sumo package", "binary": "sumo",
            "binary_version": binary_version, "client_version": None}


def test_fallback_names_the_pip_package_at_the_resolved_sumo_version(monkeypatch):
    message = _reload_without_libsumo(monkeypatch, _resolution("1.27.1"))
    assert _sumo_lib.SUMO_BINDING == "traci"
    assert "pip install libsumo==1.27.1" in message
    assert "pixi install" in message
    assert "OPENCDA_SUMO_LIB=traci" in message
    assert message.isascii() and "\n" not in message


def test_fallback_without_a_known_version_still_names_the_package(monkeypatch):
    message = _reload_without_libsumo(monkeypatch, _resolution(None))
    assert "pip install libsumo==<your SUMO version>" in message
    assert message.isascii()
