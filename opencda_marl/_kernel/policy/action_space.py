"""Action-space schema validation and route-provider config derivation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


ACTION_SPACE_MODES = {"speed_only", "path_speed"}
#: The four historical selectors, kept as names for the built-in providers.
#: ``path.provider`` names any registered ``intent_provider`` instead, which
#: is what lets a plugin own its path semantics without editing the engine.
PATH_SELECTORS = {"learned", "rule_based", "random_legal", "keep_current"}
#: The built-in semantic head spans left / straight / right / keep. A
#: registered provider declares its own width through ``HeadSpec``
#: (``opencda_marl.api.path_action``); the engine never guesses it.
SEMANTIC_PATH_ACTIONS = 4
_SELECTOR_PROVIDERS = {
    "keep_current": "tao_fixed",
    "random_legal": "naive_random",
    "rule_based": "rule_based_path",
    "learned": "marl",
}


@dataclass(frozen=True)
class PathActionConfig:
    selector: str = "keep_current"
    reward_dim: int = 0
    preference_dim: int = 0
    force_intent: str | None = None
    head: dict[str, Any] | None = None
    #: The registered ``intent_provider`` this path action resolves to, and
    #: the whole ``path`` block as the provider's own config. A provider
    #: reads its keys from ``options``; the engine reads none of them.
    provider: str = "marl"
    options: dict = None
    #: When the one-shot decision is taken: ``spawn`` (historical) or
    #: ``approach`` — inside a decision zone at the junction, on fresh
    #: observations (see _kernel/policy/decision_zone.py).
    decision_trigger: str = "spawn"
    decision_distance_m: float = 30.0

    @property
    def num_actions(self) -> int:
        return SEMANTIC_PATH_ACTIONS


@dataclass(frozen=True)
class ActionSpaceConfig:
    mode: str
    path: PathActionConfig

    @property
    def uses_path(self) -> bool:
        return self.mode == "path_speed"

    @property
    def uses_learned_path(self) -> bool:
        return self.uses_path and self.path.selector == "learned"


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"expected mapping, got {type(value).__name__}")


def resolve_action_space_config(marl_cfg: Mapping[str, Any]) -> ActionSpaceConfig:
    """Validate the action-space schema.

    ``MARL.intent_policy`` is intentionally rejected. The route/path action
    lifecycle is selected by ``MARL.action_space.mode`` and
    ``MARL.action_space.path.selector``.
    """
    if "intent_policy" in marl_cfg:
        raise ValueError(
            "MARL.intent_policy is no longer supported in active configs. "
            "Use MARL.action_space.mode and MARL.action_space.path instead."
        )
    raw = _as_dict(marl_cfg.get("action_space"))
    if not raw:
        raise ValueError(
            "MARL.action_space is required. Use mode=speed_only or mode=path_speed."
        )
    mode = str(raw.get("mode", "")).lower()
    if mode not in ACTION_SPACE_MODES:
        raise ValueError(
            f"Unsupported MARL.action_space.mode={mode!r}. "
            f"Expected one of {sorted(ACTION_SPACE_MODES)}."
        )
    path_raw = _as_dict(raw.get("path"))
    explicit_provider = path_raw.get("provider")
    selector = str(
        path_raw.get("selector", "keep_current" if mode == "speed_only" else "")
    ).lower()
    if mode == "path_speed" and not explicit_provider and selector not in PATH_SELECTORS:
        raise ValueError(
            f"Unsupported MARL.action_space.path.selector={selector!r}. "
            f"Expected one of {sorted(PATH_SELECTORS)}, or name a registered "
            "intent_provider with MARL.action_space.path.provider."
        )
    if mode == "speed_only":
        selector = "keep_current"
        explicit_provider = None
    # An explicit provider wins; otherwise the historical selector names one.
    provider = str(explicit_provider) if explicit_provider else _SELECTOR_PROVIDERS.get(
        selector, "tao_fixed")
    if explicit_provider and selector not in PATH_SELECTORS:
        # A provider named without a built-in selector drives the learned
        # lifecycle: it owns the head. Naming a built-in selector next to it
        # (e.g. ``selector: rule_based``) runs the provider with no learned
        # head.
        selector = "learned"
    pref_cfg = _as_dict(path_raw.get("preference"))
    head_cfg = _as_dict(path_raw.get("head"))
    zone_raw = _as_dict(path_raw.get("decision_zone"))
    # A profile written for a path action the engine no longer defines
    # must fail here, loudly, rather than silently run the semantic one.
    if str(path_raw.get("interface", "semantic")).lower() != "semantic":
        raise ValueError(
            f"MARL.action_space.path.interface={path_raw.get('interface')!r} is not "
            "supported: the built-in path action is the semantic turn action. A "
            "different path action is a registered intent_provider, selected with "
            "MARL.action_space.path.provider.")
    return ActionSpaceConfig(
        mode=mode,
        path=PathActionConfig(
            selector=selector,
            reward_dim=int(path_raw.get("reward_dim", pref_cfg.get("dim", 0)) or 0),
            preference_dim=int(pref_cfg.get("dim", 0) or 0),
            force_intent=path_raw.get("force_intent"),
            head=head_cfg,
            provider=provider,
            options=dict(path_raw),
            decision_trigger=str(zone_raw.get("trigger", "spawn")).lower(),
            decision_distance_m=float(zone_raw.get("distance_m", 30.0) or 30.0),
        ),
    )


def with_baseline_default(marl_cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    """A ``MARL`` block for a baseline agent implies ``action_space: speed_only``.

    Baseline scenarios (``rule_based`` / ``vanilla`` / ``behavior``) configure
    no learner and therefore no action space: the route hook is off and the
    agents drive themselves. ``MARLManager`` and ``derive_route_intent_config``
    (the scenario manager's path, built first) both apply that default, so a
    baseline config is never rejected for lacking an action space. Learner
    configs keep the strict contract: a configured algorithm without an
    action space is still an error.
    """
    if (marl_cfg.get("algorithm") is None
            and "action_space" not in marl_cfg
            and "intent_policy" not in marl_cfg):
        merged = dict(marl_cfg)
        merged["action_space"] = {"mode": "speed_only"}
        return merged
    return marl_cfg


def derive_route_intent_config(
    agents_cfg: Mapping[str, Any],
    marl_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the low-level route-intent provider config from action_space.

    The traffic manager still consumes a ``route_intent`` block for per-vehicle
    preference metadata and built-in provider knobs. The action-space block is
    the source of truth for whether the route hook is disabled, rule-based, or
    learned. The spawn-time preference sampler lives canonically under
    ``agents.preference`` and is folded into the derived block here; the
    historical ``agents.route_intent.preference`` location still works.
    """
    route_cfg = _as_dict(agents_cfg.get("route_intent"))
    # Canonical key wins when both locations are present.
    if agents_cfg.get("preference") is not None:
        route_cfg["preference"] = agents_cfg["preference"]
    action_space = resolve_action_space_config(with_baseline_default(marl_cfg))
    if action_space.mode == "speed_only":
        route_cfg["mode"] = "disabled"
    elif action_space.path.selector == "learned":
        route_cfg["mode"] = "marl"
    elif action_space.path.selector == "rule_based":
        route_cfg["mode"] = "rule_based_path"
    elif action_space.path.selector == "random_legal":
        route_cfg["mode"] = "naive_random"
    elif action_space.path.selector == "keep_current":
        route_cfg["mode"] = "tao_fixed"
    else:  # pragma: no cover - validated above
        raise ValueError(f"unsupported path selector: {action_space.path.selector}")
    route_cfg["decision_zone"] = {
        "trigger": action_space.path.decision_trigger,
        "distance_m": action_space.path.decision_distance_m,
    }
    if action_space.mode != "speed_only" and action_space.path.provider not in (
            "tao_fixed", "naive_random", "rule_based_path", "marl"):
        # A registered provider: it is the mode, and its whole path block is
        # its config.
        route_cfg["mode"] = action_space.path.provider
    # The provider's own keys, so a plugin configures itself from one block.
    for key, value in (action_space.path.options or {}).items():
        route_cfg.setdefault(key, value)
    return route_cfg
