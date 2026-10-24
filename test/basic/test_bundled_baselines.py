"""The bundled RAL baselines: present, intact, paired, and loadable.

Each entry the manifest marks ``in_wheel`` must exist with the recorded
SHA-256, name a bundled recording, and load strictly into today's learner
through the framework's own factory and load path (``_pretrained`` set by
the loader). The two TD3-44d entries are release assets and only appear
in the manifest.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from opencda_marl._bundled import (BUNDLED_CHECKPOINTS, BUNDLED_CONFIG_ROOT,
                                   bundled_checkpoint, bundled_recording)
from opencda_marl._kernel.policy.manager import MARLManager
from opencda_marl.tools.config_lint import merged_config

REPO = Path(__file__).resolve().parents[2]
MANIFEST = json.loads((BUNDLED_CHECKPOINTS / "ral" / "manifest.json").read_text(encoding="utf-8"))
IN_WHEEL = MANIFEST["entries"]
_IDS = [e["run"] for e in IN_WHEEL]


def _manager(profile_ref: str) -> MARLManager:
    cfg = dict(merged_config(REPO, profile_ref, BUNDLED_CONFIG_ROOT)["MARL"])
    cfg["outputs"] = {"metrics_store": {"enabled": False}, "tensorboard": {"enabled": False}}
    cfg.setdefault("training", {})["save_checkpoints"] = False
    return MARLManager(cfg, cfg["algorithm"])


def test_kit_shape():
    """Seven bundled entries; the two undistributed ones are recorded, not linked."""
    assert len(MANIFEST["entries"]) == 7
    assert not any("44d" in e["run"] for e in MANIFEST["entries"])
    not_distributed = MANIFEST["not_distributed"]["runs"]
    assert {e["run"] for e in not_distributed} == {"td3_44d_300vph", "td3_44d_400vph"}
    # Their configuration ships even though their weights do not.
    for entry in not_distributed:
        assert (BUNDLED_CONFIG_ROOT / f"{entry['profile']}.yaml").is_file()


@pytest.mark.parametrize("entry", IN_WHEEL, ids=_IDS)
def test_bundled_checkpoint_is_intact_and_paired(entry):
    path = bundled_checkpoint(f"ral/{entry['run']}")
    assert path is not None and path.is_file()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    assert bundled_recording(entry["recording"]) is not None
    assert (BUNDLED_CONFIG_ROOT / f"{entry['profile']}.yaml").is_file()


@pytest.mark.parametrize("entry", IN_WHEEL, ids=_IDS)
def test_bundled_checkpoint_loads_into_todays_learner(entry):
    manager = _manager(entry["profile"])
    manager.load_checkpoint(str(bundled_checkpoint(f"ral/{entry['run']}")))
    assert getattr(manager.algorithm, "_pretrained", None) is True


def test_resolver_rejects_escapes():
    assert bundled_checkpoint("../secrets.pth") is None
    assert bundled_checkpoint("ral/does_not_exist") is None


@pytest.mark.parametrize("env_cls_path", [
    "opencda_marl._kernel.envs.sumo_marl_env:SumoMARLEnv",
    "opencda_marl._kernel.envs.marl_env:MARLEnv",
])
def test_env_config_path_resolves_a_kit_name(env_cls_path):
    """``load_checkpoint: ral/<run>`` reaches the manager as the packaged file."""
    import importlib
    from types import SimpleNamespace

    module_name, cls_name = env_cls_path.split(":")
    env = object.__new__(getattr(importlib.import_module(module_name), cls_name))
    calls: list[str] = []
    env.marl_manager = SimpleNamespace(load_checkpoint=lambda p: calls.append(p))
    env._load_checkpoint_from_config("ral/sac_300vph")
    assert calls == [str(bundled_checkpoint("ral/sac_300vph"))]

    with pytest.raises(FileNotFoundError):
        env._load_checkpoint_from_config("ral/not_a_run")
