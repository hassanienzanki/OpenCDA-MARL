"""Algorithm construction for the canonical MARL manager.

Construction is data-driven: the registry's spec carries the factory, so
this module never names a concrete algorithm class. Adding an algorithm
touches the registry only (see ``registry.register_algorithm``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from .manager import MARLManager


def build_algorithm(manager: "MARLManager", algorithm: str | None):
    """Build the configured algorithm object through the canonical registry."""
    if algorithm is None:
        return None

    spec = manager.algorithm_spec
    if spec is None:
        raise ValueError("Algorithm spec was not initialized")

    algo_config = manager._merge_shared_config(
        manager.config.get(spec.config_section, {})
    )

    # Vector-critic families learn one value per objective; scalar families
    # must not receive the key at all (they would silently reinterpret it).
    if (
        spec.vector_rewards
        and manager.reward_mode == "objective_vector"
        and manager.objective_spec is not None
    ):
        algo_config["reward_dim"] = len(manager.objective_spec.objectives)

    state_dim = manager.resolve_state_dim()
    action_dim = manager.config.get("action_dim", 1)

    policy = spec.build(algo_config, state_dim=state_dim, action_dim=action_dim)
    logger.debug(
        f"Built {spec.label} via registry "
        f"(state_dim={state_dim}, action_dim={action_dim})")
    return policy
