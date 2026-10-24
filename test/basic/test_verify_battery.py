"""The verification battery's SUMO smoke profile (scripts/verify.py).

The battery is made for repeated runs, so it must leave nothing behind but
its own artifacts: a profile that kept the default ``outputs.monitor`` would
start a detached monitor for the verify root on every run, holding a port
and (on Windows) the root directory itself after the battery exits.
"""
import importlib.util
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "verify.py"

pytestmark = pytest.mark.skipif(not SCRIPT.is_file(), reason="not a source checkout")


@pytest.fixture(scope="module")
def verify():
    spec = importlib.util.spec_from_file_location("verify_battery_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def profile(verify, tmp_path):
    return yaml.safe_load(verify.sumo_profile(tmp_path / "sumo"))


def test_the_smoke_profile_writes_only_under_the_verify_root(profile, tmp_path):
    outputs = profile["outputs"]
    assert outputs["root_dir"] == (tmp_path / "sumo").as_posix()
    assert profile["MARL"]["training"]["checkpoint_dir"].startswith(outputs["root_dir"])
    assert outputs["tensorboard"]["enabled"] is False


def test_the_smoke_run_starts_no_monitor(profile, monkeypatch):
    """Exercised through the coordinator's own switch, not just the key."""
    import opencda_marl.telemetry.monitor as monitor
    from opencda_marl._kernel.lifecycle.coordinator import MARLCoordinator

    calls = []
    monkeypatch.setattr(monitor, "ensure_service",
                        lambda root, host, port, allow_hosts=(): calls.append(root) or ("http://x", True))
    MARLCoordinator({"outputs": profile["outputs"]})._setup_monitor_service()
    assert calls == [], "the verify profile must disable outputs.monitor"


def test_the_carla_profile_writes_only_under_the_verify_root_and_starts_no_monitor(verify, tmp_path):
    cfg = yaml.safe_load(verify.carla_profile(tmp_path / "carla"))
    assert cfg["outputs"]["root_dir"] == (tmp_path / "carla").as_posix()
    assert cfg["outputs"]["monitor"] == {"enabled": False}
    assert cfg["outputs"]["tensorboard"]["enabled"] is False
    # Still the bundled baseline: same agent type and episode count.
    assert cfg["agents"]["agent_type"] == "rule_based"
    assert cfg["scenario"]["simulation"]["max_episodes"] == 1
