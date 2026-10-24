"""Path-action lifecycle wiring for the canonical MARL manager."""

from __future__ import annotations

from loguru import logger


def configure_path_action(manager) -> None:
    """Attach a learned one-shot path head when action_space requests it."""
    if not manager.action_space_config.uses_learned_path:
        return
    if manager.algorithm is None:
        raise ValueError("path_speed learned path requires a base MARL algorithm")

    # Eligibility is a declared capability plus a verified implementation,
    # not a hardcoded name list: an externally registered policy that
    # implements IntentPolicy works here with no edit to this file.
    from opencda_marl.api.policy import missing_intent_methods, supports_intent

    spec = manager.algorithm_spec
    if spec is not None and not spec.capabilities.supports_intent:
        raise NotImplementedError(
            f"Learned path is not implemented for {manager.algorithm_label}: the "
            "algorithm is registered with supports_intent=False. Use "
            "rule_based/random_legal/keep_current for this family, or register "
            "it with supports_intent=True once it implements IntentPolicy."
        )
    if not supports_intent(manager.algorithm):
        missing = ", ".join(missing_intent_methods(manager.algorithm))
        raise NotImplementedError(
            f"Learned path is not implemented for {manager.algorithm_label}: "
            f"the policy is missing IntentPolicy method(s): {missing}."
        )

    from opencda_marl._kernel.policy.accumulator import PerActorRewardAccumulator
    from opencda_marl.api.intent_observation import SPAWN_OBS_DIM_NO_PREF
    from opencda_marl.api.path_action import resolve_head_spec

    path_cfg = manager.action_space_config.path
    pref_dim = int(path_cfg.preference_dim)
    reward_dim = int(path_cfg.reward_dim)
    if pref_dim <= 0 or reward_dim <= 0:
        raise ValueError(
            "Learned path action requires action_space.path.preference.dim "
            "and action_space.path.reward_dim to be positive."
        )
    if reward_dim > 0 and manager.objective_spec is not None:
        objective_dim = len(manager.objective_spec.objectives)
        if reward_dim != objective_dim:
            raise ValueError(
                f"Path reward_dim={reward_dim} does not match objective_dim={objective_dim}. "
                "The learned path head trains on the run's objective vector; set "
                "action_space.path.reward_dim to its width."
            )
    head_cfg = manager._merge_shared_config(dict(path_cfg.head or {}))
    head_cfg["preference"] = {"dim": pref_dim}
    head_cfg["reward_dim"] = reward_dim
    # The provider owns its observation and its number of alternatives; the
    # engine only appends the preference vector. A provider that declares
    # nothing keeps the built-in shape.
    spec = resolve_head_spec(path_cfg.provider, path_cfg.options)
    observation_dim = spec.observation_dim if spec is not None else SPAWN_OBS_DIM_NO_PREF
    num_actions = spec.num_actions if spec is not None else path_cfg.num_actions
    state_dim = observation_dim + pref_dim
    manager.algorithm.enable_route_intent(
        head_cfg,
        state_dim=state_dim,
        num_actions=num_actions,
        pref_dim=pref_dim,
        reward_dim=reward_dim,
    )
    manager.intent_algorithm = manager.algorithm
    manager.intent_reward_accumulator = PerActorRewardAccumulator(reward_dim=reward_dim)
    manager.intent_reward_scale = float(head_cfg.get("reward_scale", 1.0) or 1.0)
    manager.intent_update_interval_steps = max(
        1,
        int(head_cfg.get("update_interval_steps", 1) or 1),
    )
    logger.success(
        f"Learned path head enabled for {manager.algorithm_label}: "
        f"state_dim={state_dim}, pref_dim={pref_dim}, reward_dim={reward_dim}, "
        f"reward_scale={manager.intent_reward_scale}, "
        f"update_interval_steps={manager.intent_update_interval_steps}"
    )
