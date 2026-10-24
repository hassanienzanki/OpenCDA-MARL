"""External algorithms: one registration path, four discovery routes.

Built-ins and plugins go through the same ``register_algorithm`` call;
learners are found through the package entry point, the ``contrib/``
drop-in folder, ``OPENCDA_MARL_PLUGINS``, or ``MARL.plugins``; a failed
import is reported, not hidden; a plugin targeting another policy-API
version is rolled back; a plain ``Policy`` (no BaseAlgorithm) survives the
world-reload snapshot and gets a manager-owned metrics store.
"""
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CONTRIB_MODULE = "opencda_marl.builtin.algorithms.contrib.random_speed"


@pytest.fixture
def registry():
    import importlib

    reg = importlib.import_module("opencda_marl.api.algorithms")
    from opencda_marl.plugins import loader as plugin_loader
    before = (dict(reg._SPECS), dict(reg._SOURCES),
              dict(plugin_loader._plugin_failures), plugin_loader._plugins_loaded,
              set(plugin_loader._loaded_extras))
    yield reg
    reg._SPECS.clear(); reg._SPECS.update(before[0])
    reg._SOURCES.clear(); reg._SOURCES.update(before[1])
    plugin_loader._plugin_failures.clear(); plugin_loader._plugin_failures.update(before[2])
    plugin_loader._plugins_loaded = before[3]
    plugin_loader._loaded_extras.clear(); plugin_loader._loaded_extras.update(before[4])
    for name in [m for m in sys.modules if m.startswith("opencda_marl_plugin_")]:
        sys.modules.pop(name, None)


def _plugin_source(key: str, api_line: str = "") -> str:
    return textwrap.dedent(f"""
        {api_line}
        from opencda_marl.api.algorithms import register_algorithm

        @register_algorithm("{key}", label="{key.title()}")
        def build(config, *, state_dim, action_dim):
            return object()
        """)


def test_builtins_are_registered_through_the_public_api(registry):
    rows = {r["key"]: r for r in registry.describe_algorithms()}
    assert set(rows) >= {"matd3", "pcmatd3", "masac", "pcmasac", "mappo", "pcmappo", "dqn", "q_learning"}
    assert all(rows[k]["source"] == "builtin" for k in ("masac", "pcmappo", "dqn"))
    assert rows["pcmasac"]["reward_aggregation"] == "team_mean" and rows["pcmasac"]["supports_intent"]
    assert rows["matd3"]["supports_intent"] is False
    # the legacy private table is gone: the registry module has no _lazy helper
    assert not hasattr(registry, "_lazy")


def test_contrib_drop_in_is_discovered_automatically(registry):
    """A module dropped into opencda_marl.builtin.algorithms/contrib/ registers at
    package load with no entry point, env var, or config edit."""
    registry.load_plugins()
    spec = registry.resolve_algorithm_spec("random_speed")
    assert spec.label == "RandomSpeed"
    assert "opencda_marl.builtin.algorithms" in registry._SOURCES["random_speed"]
    policy = spec.build({"min_action": 10, "max_action": 20, "seed": 1}, state_dim=5, action_dim=1)
    action = policy.select_action({"v0": None}, "v0")
    assert 10 <= action <= 20


def test_plugin_by_file_path_via_environment(registry, monkeypatch, tmp_path):
    module = tmp_path / "env_algo.py"
    module.write_text(_plugin_source("env_algo"), encoding="utf-8")
    monkeypatch.setenv(registry.PLUGIN_ENV, str(module))
    failures = registry.load_plugins(force=True)
    assert failures == {}
    assert "env_algo" in registry.registered_algorithm_keys()
    assert registry._SOURCES["env_algo"] == str(module)


def test_plugin_by_module_name_via_config_list(registry, monkeypatch, tmp_path):
    (tmp_path / "cfg_algo.py").write_text(_plugin_source("cfg_algo"), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    failures = registry.load_plugins(["cfg_algo"])
    assert failures == {}
    assert "cfg_algo" in registry.registered_algorithm_keys()
    sys.modules.pop("cfg_algo", None)


def test_broken_plugin_is_reported_in_the_unknown_algorithm_error(registry, tmp_path):
    bad = tmp_path / "bad_plugin.py"
    bad.write_text("raise ImportError('boom: missing torch_extra')\n", encoding="utf-8")
    failures = registry.load_plugins([str(bad)])
    assert str(bad) in failures and "boom" in failures[str(bad)]
    with pytest.raises(ValueError, match="Plugins that failed to import") as err:
        registry.resolve_algorithm_spec("definitely_not_registered")
    assert "boom" in str(err.value)


def test_policy_api_mismatch_is_rolled_back(registry, tmp_path):
    """A plugin targeting another contract version must not stay registered."""
    future = tmp_path / "future_algo.py"
    future.write_text(_plugin_source("future_algo", "OPENCDA_MARL_POLICY_API = 999"),
                      encoding="utf-8")
    failures = registry.load_plugins([str(future)])
    assert "plugin API v999" in failures[str(future)]
    assert "future_algo" not in registry.registered_algorithm_keys()

    current = tmp_path / "current_algo.py"
    current.write_text(
        _plugin_source("current_algo", f"OPENCDA_MARL_POLICY_API = {registry.POLICY_API_VERSION}"),
        encoding="utf-8")
    failures = registry.load_plugins([str(current)])
    assert str(current) not in failures          # cumulative dict still lists future_algo
    assert "current_algo" in registry.registered_algorithm_keys()


def test_registering_over_a_builtin_requires_override(registry):
    registry.load_plugins()
    with pytest.raises(ValueError, match="already registered"):
        registry.register_algorithm("masac")(lambda config, *, state_dim, action_dim: None)
    registry.register_algorithm("masac", override=True, source="test")(
        lambda config, *, state_dim, action_dim: "replaced")
    assert registry.resolve_algorithm_spec("masac").build({}, state_dim=1, action_dim=1) == "replaced"


def test_manager_owns_a_store_for_plain_policies(registry, tmp_path):
    from opencda_marl._kernel.policy.manager import MARLManager
    from opencda_marl.telemetry import MetricsStore

    cfg = {
        "algorithm": "random_speed",
        "action_space": {"mode": "speed_only"},
        "reward_profile": "competitive_intersection_v2",
        "state_dim": 3, "action_dim": 1,
        "random_speed": {"features": {"rel_x": 1, "rel_y": 1, "speed": 1},
                         "min_action": 5.0, "max_action": 9.0, "seed": 3},
        "outputs": {"metrics_store": {"enabled": True, "log_dir": str(tmp_path), "run_name": "plug",
                                      "log_frequency": 1}},
    }
    manager = MARLManager(cfg, "random_speed")
    assert manager.algorithm.__class__.__name__ == "RandomSpeedPolicy"
    assert manager._external_store is not None, "plain policies get a manager-owned store"
    manager.record_update_metrics(manager.algorithm.update())
    manager.record_update_metrics(manager.algorithm.update())
    manager.reset_episode(episode_metrics={"total_reward": -3.0, "step_length": 12, "success_rate": 50.0,
                                           "collision_rate": 0.0})
    manager.close()
    reader = MetricsStore(tmp_path / "random_speed" / "plug", read_only=True)
    try:
        tags = {t["tag"] for t in reader.tags()}
        assert "random_speed/updates" in tags and "Episode/reward" in tags, tags
        eps = reader.episodes()
        assert eps[0]["reward"] == -3.0 and eps[0]["length"] == 12.0 and eps[0]["success_rate"] == 50.0
        assert reader.meta()["external_policy"] is True
    finally:
        reader.close()


def test_world_snapshot_tolerates_a_plain_policy(registry):
    """TrainingStateSnapshot read training_step / episode_count / writer /
    is_converged unguarded -- an external policy crashed the first CARLA
    world reload."""
    import importlib
    from types import SimpleNamespace

    from opencda_marl._kernel.lifecycle.world_manager import TrainingStateSnapshot

    plug = importlib.import_module(CONTRIB_MODULE)
    policy = plug.RandomSpeedPolicy({"max_action": 20.0}, 3, 1)
    env = SimpleNamespace(marl_manager=SimpleNamespace(algorithm=policy), metrics=object(),
                          checkpoint_manager=object(), training_config={})
    snapshot = TrainingStateSnapshot.capture(env)
    assert snapshot.training_step == 0 and snapshot.episode_count == 0
    snapshot.restore(env)                       # must not add attributes the policy never had
    assert not hasattr(policy, "training_step") and not hasattr(policy, "is_converged")


def test_external_config_root_inherits_default_and_base(tmp_path):
    """A package may ship only a profile; family default and scenario base
    come from the repository (or bundled) roots."""
    import shutil

    from opencda_marl.utils.config_locator import resolve_marl_config_paths

    external = tmp_path / "configs" / "sumo-marl" / "intersection"
    external.mkdir(parents=True)
    shutil.copyfile(REPO / "opencda_marl/_bundled/configs/sumo-marl/intersection/random_speed.yaml",
                    external / "my_profile.yaml")
    resolved = resolve_marl_config_paths(REPO, "sumo-marl/intersection/my_profile",
                                         custom_config_roots=tmp_path / "configs")
    assert resolved["scenario_yaml"].resolve() == (external / "my_profile.yaml").resolve()
    assert resolved["default_yaml"].resolve() == (REPO / "opencda_marl/_bundled/configs/sumo-marl/default.yaml").resolve()
    assert resolved["scenario_base_yaml"].resolve() == (REPO / "opencda_marl/_bundled/configs/sumo-marl/intersection/base.yaml").resolve()
