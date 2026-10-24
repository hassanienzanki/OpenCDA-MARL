from __future__ import annotations


def has_pending_sumo_traffic(min_expected_number: int, pending_events: int = 0) -> bool:
    return int(min_expected_number) > 0 or int(pending_events) > 0


def should_end_sumo_episode(
    current_step: int,
    max_steps: int,
    has_observations: bool,
    min_expected_number: int,
    pending_events: int = 0,
) -> bool:
    if current_step >= max_steps:
        return True
    if has_observations:
        return False
    return not has_pending_sumo_traffic(min_expected_number, pending_events)


def build_sumo_episode_states(
    current_step: int,
    collision_count: int,
    success_count: int,
    timeout_count: int,
    step_length: float,
) -> dict:
    return {
        'step': current_step,
        'collision': collision_count,
        'success': success_count,
        'active_agents': timeout_count,
        'fixed_dt': step_length,
    }
