"""The matrix launcher finds CARLA only where it is told to.

``resolve_carla_binary`` reads an explicit binary or root (arguments, then
``CARLA_BIN`` / ``CARLA_ROOT``) and raises when it has neither, instead of
guessing an install path that only fails later, when a crashed server cannot
be restarted. The launcher needs a binary only in ``--carla-restart-mode
process`` outside a dry run, so a dry run and the docker / none modes work
with no CARLA configured at all.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts" / "experiments"
LAUNCHER = SCRIPTS / "launch_matrix.py"
_CARLA_ENV = ("CARLA_BIN", "CARLA_ROOT", "CARLA_HOME")


def _matrix_common():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import importlib
    return importlib.import_module("matrix_common")


@pytest.fixture
def no_carla_env(monkeypatch):
    for name in _CARLA_ENV:
        monkeypatch.delenv(name, raising=False)


def test_without_a_binary_or_root_it_raises(no_carla_env):
    mc = _matrix_common()
    with pytest.raises(ValueError, match="CARLA_ROOT"):
        mc.resolve_carla_binary()


def test_a_root_names_the_platform_launcher(no_carla_env, tmp_path):
    mc = _matrix_common()
    expected = "CarlaUE4.exe" if platform.system().lower().startswith("win") else "CarlaUE4.sh"
    assert mc.resolve_carla_binary(carla_root=str(tmp_path)) == tmp_path / expected


def test_the_environment_is_read_and_an_explicit_binary_wins(no_carla_env, monkeypatch, tmp_path):
    mc = _matrix_common()
    monkeypatch.setenv("CARLA_ROOT", str(tmp_path / "root"))
    assert mc.resolve_carla_binary().parent == tmp_path / "root"
    monkeypatch.setenv("CARLA_BIN", str(tmp_path / "bin" / "carla"))
    assert mc.resolve_carla_binary() == tmp_path / "bin" / "carla"
    assert mc.resolve_carla_binary(carla_bin=str(tmp_path / "x")) == tmp_path / "x"


def _run_launcher(*args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in _CARLA_ENV}
    return subprocess.run(
        [sys.executable, str(LAUNCHER), "--conditions", "speed_only_pcmappo",
         "--regimes", "aligned", "--seeds", "0", *args],
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=120,
    )


def test_a_dry_run_needs_no_carla():
    result = _run_launcher("--dry-run")      # default restart mode: process
    assert result.returncode == 0, result.stderr[-1500:]
    assert "CARLA binary: not used" in result.stdout


def test_process_mode_without_a_binary_stops_before_any_cell():
    result = _run_launcher("--carla-restart-mode", "process")
    assert result.returncode != 0
    assert "CARLA_ROOT" in result.stderr
    assert "launching" not in result.stdout, "no cell may start without a way to restart CARLA"
