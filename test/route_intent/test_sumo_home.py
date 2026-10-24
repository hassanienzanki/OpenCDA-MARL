"""SUMO install resolution: one consistent client + binary."""
import os
import sys
from pathlib import Path

import pytest

from opencda_marl._kernel.traffic import _sumo_home


def test_resolver_prefers_the_environment_package_and_records_versions():
    info = _sumo_home.resolve_sumo_home()
    assert info["source"] in ("eclipse-sumo package", "SUMO_HOME")
    assert Path(info["binary"]).exists(), info
    # When the package is present, SUMO_HOME must now point at it so any
    # later sumolib.checkBinary / tools import agrees with the client.
    if info["source"] == "eclipse-sumo package":
        assert os.environ["SUMO_HOME"] == info["home"]
        assert not any(
            p.endswith(os.path.join("Eclipse", "Sumo", "tools")) for p in sys.path
        ), "a stale system SUMO_HOME/tools entry survived on sys.path"


def test_binding_and_binary_agree_on_major_minor():
    info = _sumo_home.resolve_sumo_home()
    if not (info["binary_version"] and info["client_version"]):
        pytest.skip("version probing unavailable")
    assert info["binary_version"].split(".")[:2] == info["client_version"].split(".")[:2], info


def test_sumo_binary_paths_exist():
    assert Path(_sumo_home.sumo_binary(gui=False)).exists()
    assert Path(_sumo_home.sumo_binary(gui=True)).exists()


def test_env_module_imports_without_sumo_home(monkeypatch):
    """The env used to sys.exit() at import if SUMO_HOME was unset."""
    monkeypatch.delenv("SUMO_HOME", raising=False)
    # Force a fresh resolution under the modified environment.
    monkeypatch.setattr(_sumo_home, "_RESOLVED", None)
    info = _sumo_home.resolve_sumo_home()
    assert info["source"] == "eclipse-sumo package"
    import importlib
    import opencda_marl._kernel.envs.sumo_marl_env as env_mod
    importlib.reload(env_mod)  # must not raise / exit
