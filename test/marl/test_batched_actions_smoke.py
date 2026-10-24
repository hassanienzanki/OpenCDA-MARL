"""Smoke test for the unified batched-action template-method refactor.

Validates that PCMAPPO, MAPPO, SACAlgorithm (as PCMASAC), and TD3
all properly opt into the shared ``BaseAlgorithm.select_actions``
dispatcher and produce sensible per-agent action dicts.

For each algorithm:

1. ``supports_batched_actions`` is True.
2. ``select_actions(obs, ids, training=False)`` returns one entry per id.
3. Each speed (when not None for TD3 warmup) is a Python float inside
   ``[min_action, max_action]``.
4. PPO families (MAPPO, PCMAPPO) populate per-agent ``last_log_probs``
   and value dicts as a side effect (so ``store_transition`` can read
   them next step without ``KeyError``).
5. The fallback path engages cleanly when the batched forward is forced
   to raise (we monkey-patch ``_batched_forward`` to error).

Run with: ``pixi run python -m pytest test/marl/test_batched_actions_smoke.py -x -v``
or as a script: ``pixi run python test/marl/test_batched_actions_smoke.py``.
"""
from __future__ import annotations

import sys
import traceback
from typing import Dict, List

import numpy as np


def _make_obs(n_agents: int, state_dim: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed=42)
    obs = {}
    for i in range(n_agents):
        vec = rng.standard_normal(state_dim).astype(np.float32)
        # Index 6 is dist_to_intersection in the TD3 feature schema; keep it
        # in a sane positive range so the distance-bucketing branches all
        # get exercised across the 3 agents.
        vec[6] = float(5.0 + 10.0 * i)
        obs[str(i)] = vec
    return obs


def _check_speeds(speeds: Dict[str, float], ids: List[str], min_a: float, max_a: float, allow_none: bool = False):
    assert set(speeds.keys()) == set(ids), f"keys {set(speeds.keys())} vs {set(ids)}"
    for aid, sp in speeds.items():
        if sp is None:
            assert allow_none, f"agent {aid} got None speed (only allowed in TD3 warmup)"
            continue
        assert isinstance(sp, float), f"speed for {aid} is {type(sp).__name__}, expected float"
        assert min_a <= sp <= max_a, f"speed {sp} out of [{min_a}, {max_a}]"


def _check_supports(algo) -> None:
    assert getattr(algo, "supports_batched_actions", False) is True, (
        f"{type(algo).__name__}.supports_batched_actions must be True"
    )


def _check_fallback(algo, obs, ids):
    """Force the batched forward to raise and confirm the dispatcher falls
    back to per-agent ``select_action`` without crashing."""
    original = algo._batched_forward
    def _boom(*a, **k):
        raise RuntimeError("forced batched-path failure")
    algo._batched_forward = _boom
    try:
        result = algo.select_actions(obs, ids, training=False)
    finally:
        algo._batched_forward = original
    assert set(result.keys()) == set(ids), "fallback dict missing keys"


def test_pcmappo():
    from opencda_marl.builtin.algorithms.pcmappo import PCMAPPOAlgorithm
    cfg = {
        "actor_hidden_dims": [64, 64],
        "critic_hidden_dims": [64, 64],
        "lstm_hidden_size": 32,
        "lstm_num_layers": 1,
        "pref_dim": 2,
        "reward_dim": 2,
        "max_action": 45.0,
        "min_action": 0.0,
        "warmup_steps": 0,
        "batch_size": 8,
        "share_policy": True,
    }
    algo = PCMAPPOAlgorithm(cfg, state_dim=11, action_dim=1)
    _check_supports(algo)

    obs = _make_obs(3, 11)
    ids = list(obs.keys())
    speeds = algo.select_actions(obs, ids, training=False)
    _check_speeds(speeds, ids, 0.0, 45.0)

    # PPO stash side-effect must be populated.
    for aid in ids:
        assert aid in algo.last_log_probs, f"PCMAPPO last_log_probs missing {aid}"
        assert aid in algo.last_scalar_values, f"PCMAPPO last_scalar_values missing {aid}"

    _check_fallback(algo, obs, ids)
    print("[ok] PCMAPPO: supports + batched + stash + fallback")


def test_mappo():
    from opencda_marl.builtin.algorithms.mappo import MAPPOAlgorithm
    cfg = {
        "actor_hidden_dims": [64, 64],
        "critic_hidden_dims": [64, 64],
        "lstm_hidden_size": 32,
        "lstm_num_layers": 1,
        "max_action": 45.0,
        "min_action": 0.0,
        "warmup_steps": 0,
        "batch_size": 8,
        "share_policy": True,
    }
    algo = MAPPOAlgorithm(cfg, state_dim=9, action_dim=1)
    _check_supports(algo)

    obs = _make_obs(3, 9)
    ids = list(obs.keys())
    speeds = algo.select_actions(obs, ids, training=False)
    _check_speeds(speeds, ids, 0.0, 45.0)

    for aid in ids:
        assert aid in algo.last_log_probs, f"MAPPO last_log_probs missing {aid}"
        assert aid in algo.last_values, f"MAPPO last_values missing {aid}"

    _check_fallback(algo, obs, ids)
    print("[ok] MAPPO: supports + batched + stash + fallback")


def test_sac():
    from opencda_marl.builtin.algorithms.sac import SACAlgorithm
    cfg = {
        "actor_hidden_dims": [64, 64],
        "critic_hidden_dims": [64, 64],
        "lstm_hidden_size": 32,
        "lstm_num_layers": 1,
        "pref_dim": 2,
        "reward_dim": 2,
        "max_action": 45.0,
        "min_action": 0.0,
        "warmup_steps": 0,
        "memory_size": 1000,
        "batch_size": 32,
    }
    algo = SACAlgorithm(cfg, state_dim=11, action_dim=1)
    _check_supports(algo)

    obs = _make_obs(3, 11)
    ids = list(obs.keys())
    speeds = algo.select_actions(obs, ids, training=False)
    _check_speeds(speeds, ids, 0.0, 45.0)

    _check_fallback(algo, obs, ids)
    print("[ok] SAC: supports + batched + fallback")


def test_td3():
    from opencda_marl.builtin.algorithms.td3 import TD3Algorithm
    cfg = {
        "lstm_hidden_size": 32,
        "lstm_num_layers": 1,
        "critic_hidden_dims": [64, 64],
        "max_action": 45.0,
        "min_action": 0.0,
        "warmup_steps": 0,
        "memory_size": 1000,
        "batch_size": 32,
    }
    algo = TD3Algorithm(cfg, state_dim=8, action_dim=1)
    _check_supports(algo)

    # TD3 ego features use index 6 = dist_to_intersection.
    obs = _make_obs(3, 8)
    ids = list(obs.keys())
    algo._pretrained = True  # bypass warmup-None for the speed-bounds check
    speeds = algo.select_actions(obs, ids, training=False)
    _check_speeds(speeds, ids, 0.0, 45.0, allow_none=False)
    print("[ok] TD3: supports + batched (eval-deterministic)")

    # Training mode with distance-based noise must also stay bounded.
    speeds_train = algo.select_actions(obs, ids, training=True)
    _check_speeds(speeds_train, ids, 0.0, 45.0, allow_none=False)
    print("[ok] TD3: supports + batched (training noise + clamp)")

    # Force-warmup path returns None per id (so manager routes to vanilla).
    algo._pretrained = False
    # Empty memory => len(memory)=0 < warmup_steps; flip warmup_steps high.
    algo.warmup_steps = 10_000
    warm = algo.select_actions(obs, ids, training=True)
    _check_speeds(warm, ids, 0.0, 45.0, allow_none=True)
    assert all(v is None for v in warm.values()), "TD3 warmup should emit None"
    print("[ok] TD3: warmup returns None per agent")


def main() -> int:
    suites = [
        ("pcmappo", test_pcmappo),
        ("mappo", test_mappo),
        ("sac", test_sac),
        ("td3", test_td3),
    ]
    failures = []
    for name, fn in suites:
        try:
            fn()
        except Exception as exc:
            traceback.print_exc()
            failures.append((name, exc))
    if failures:
        print(f"\nFAILED: {[n for n, _ in failures]}")
        return 1
    print(f"\nALL {len(suites)} ALGORITHM SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
