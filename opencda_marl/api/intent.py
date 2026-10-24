"""Route-intent provider boundary: the callable shape and its resolution.

A provider is ``IntentProvider``: a callable taking the spawn context dict
and returning a ``RouteIntentDecision`` (or None). Implementations are
plugins of kind ``intent_provider``; :func:`build_provider_from_config`
resolves the configured mode through the registry. The ``marl`` mode is
the policy bridge and resolves through the wired manager instance.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from opencda_marl.api.route_intent import RouteIntentDecision

IntentProvider = Callable[[Dict[str, Any]], Optional[RouteIntentDecision]]


def build_provider_from_config(
    intent_cfg: Optional[Dict[str, Any]],
    planner: Any,
    marl_manager: Any = None,
    training: bool = True,
    previous: Optional[IntentProvider] = None,
) -> Optional[IntentProvider]:
    """Construct the configured provider (None when mode is disabled).

    Recognized config keys::

        agents:
          route_intent:
            mode: disabled | tao_fixed | naive_random | rule_based_path | marl
            seed: int  # naive_random only

    The ``marl`` mode resolves via ``marl_manager.get_intent_provider()``
    when a manager is wired. ``previous`` is the provider being replaced
    (world reload); state that should outlive the rebuild — the
    naive-random RNG — is carried over from it.
    """
    if not intent_cfg:
        return None
    mode = str(intent_cfg.get("mode", "disabled")).lower()
    if mode in ("disabled", "off", ""):
        return None
    if planner is None:
        return None
    if mode == "marl":
        if marl_manager is None:
            return None
        return marl_manager.get_intent_provider(training=training)
    if mode in ("rule_path", "rule_based"):
        mode = "rule_based_path"
    from opencda_marl.api.components import resolve_component

    try:
        factory = resolve_component("intent_provider", mode, config=intent_cfg)
    except ValueError as exc:
        raise ValueError(f"Unknown route_intent.mode: {mode!r} ({exc})") from exc
    return factory(intent_cfg, planner, marl_manager=marl_manager,
                   training=training, previous=previous)
