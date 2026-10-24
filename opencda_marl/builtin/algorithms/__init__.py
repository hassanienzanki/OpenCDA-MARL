"""The first-party learners — a plugin that happens to ship in the wheel.

This subpackage registers its learners through
:func:`opencda_marl.api.algorithms.register_algorithm` exactly like any
third-party package, and the engine discovers it through the same
``opencda_marl.plugins`` entry-point group (declared by the
``opencda-marl`` wheel on itself). The engine never imports a learner:
it guarantees observation delivery and action application, and everything
that decides — these baselines included — arrives through the plugin door.

Learners (registry keys in parentheses):
- ``TD3Algorithm`` (matd3 / pcmatd3) — twin delayed DDPG, off-policy
- ``SACAlgorithm`` (masac / pcmasac) — soft actor-critic, off-policy
- ``MAPPOAlgorithm`` (mappo), ``PCMAPPOAlgorithm`` (pcmappo) — on-policy
- ``DQNAlgorithm`` (dqn), ``QLearningAlgorithm`` (q_learning) —
  single-agent tutorial families

Shared building blocks live beside them: ``base_algorithm`` (context
encoding, warmup, checkpoint plumbing), ``training_logger``, ``gae``,
``rollout_buffer``, ``smart_replay_buffer``, ``encoders``, ``route_head``,
``value_norm``, and the one-shot route-intent learners under ``intent/``.
Registration is import-light: learner modules (and torch) load only when a
factory is actually called.
"""
from opencda_marl.api.algorithms import register_algorithm

#: Policy-contract version this module targets (checked by the engine).
OPENCDA_MARL_POLICY_API = 1


def _lazy(target: str):
    """Factory that imports ``module:Class`` only when first constructed."""
    module_name, _, class_name = target.rpartition(".")

    def _factory(config, *, state_dim, action_dim):
        import importlib

        cls = getattr(importlib.import_module(module_name), class_name)
        return cls(config, state_dim, action_dim)

    _factory.__name__ = f"build_{class_name.lower()}"
    _factory.__module__ = module_name
    return _factory


# --- off-policy, twin critics -------------------------------------------------
register_algorithm(
    "matd3", label="MATD3", source="builtin",
    multi_agent=True, reward_aggregation="per_agent", supports_intent=False,
)(_lazy("opencda_marl.builtin.algorithms.td3.TD3Algorithm"))

register_algorithm(
    "pcmatd3", label="PCMATD3", source="builtin",
    multi_agent=True, reward_aggregation="per_agent", supports_intent=False,
)(_lazy("opencda_marl.builtin.algorithms.td3.TD3Algorithm"))

register_algorithm(
    "masac", label="MASAC", source="builtin",
    multi_agent=True, reward_aggregation="per_agent", supports_intent=True,
    vector_rewards=True,
)(_lazy("opencda_marl.builtin.algorithms.sac.SACAlgorithm"))

register_algorithm(
    "pcmasac", label="PCMASAC", source="builtin",
    multi_agent=True, reward_aggregation="team_mean", supports_intent=True,
    vector_rewards=True,
)(_lazy("opencda_marl.builtin.algorithms.sac.SACAlgorithm"))

# --- on-policy ----------------------------------------------------------------
register_algorithm(
    "mappo", label="MAPPO", source="builtin",
    multi_agent=True, reward_aggregation="per_agent", supports_intent=True,
)(_lazy("opencda_marl.builtin.algorithms.mappo.MAPPOAlgorithm"))

register_algorithm(
    "pcmappo", label="PCMAPPO", source="builtin",
    multi_agent=True, reward_aggregation="team_mean", supports_intent=True,
    vector_rewards=True,
)(_lazy("opencda_marl.builtin.algorithms.pcmappo.PCMAPPOAlgorithm"))

# --- single-agent tutorial families -------------------------------------------
# Kept registered so the tutorial scenarios run; dispatched through the
# single-agent branch via multi_agent=False.
register_algorithm(
    "dqn", label="DQN", source="builtin",
    multi_agent=False, reward_aggregation="per_agent", supports_intent=False,
)(_lazy("opencda_marl.builtin.algorithms.dqn.DQNAlgorithm"))

register_algorithm(
    "q_learning", label="Q-Learning", source="builtin",
    multi_agent=False, reward_aggregation="per_agent", supports_intent=False,
)(_lazy("opencda_marl.builtin.algorithms.q_learning.QLearningAlgorithm"))


__all__ = ["QLearningAlgorithm", "DQNAlgorithm", "TD3Algorithm",
           "MAPPOAlgorithm", "PCMAPPOAlgorithm", "SACAlgorithm"]

# Learner classes resolve lazily (PEP 562): importing this package must stay
# torch-free so registration (and `python -m opencda_marl.api.algorithms`)
# never pays the torch import.
_LAZY_CLASSES = {
    "QLearningAlgorithm": ".q_learning",
    "DQNAlgorithm": ".dqn",
    "TD3Algorithm": ".td3",
    "MAPPOAlgorithm": ".mappo",
    "PCMAPPOAlgorithm": ".pcmappo",
    "SACAlgorithm": ".sac",
}


def __getattr__(name):
    module_name = _LAZY_CLASSES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


def _load_contrib() -> None:
    """Import every module dropped into ``contrib/`` so its registrations run.

    A failing contrib module is reported and skipped — one broken drop-in
    must not take the built-ins down with it.
    """
    import importlib
    import pkgutil

    from loguru import logger

    from . import contrib

    for info in pkgutil.iter_modules(contrib.__path__):
        name = f"{__name__}.contrib.{info.name}"
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - isolate broken drop-ins
            logger.warning(f"contrib algorithm {name} failed to import: {exc}")


_load_contrib()
