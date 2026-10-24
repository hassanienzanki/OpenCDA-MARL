# An Intent Provider

An intent provider decides, per vehicle, which way it goes at the
junction. The same callable serves both decision timings: at spawn
(default), or at the junction approach when the decision zone is
configured (`MARL.action_space.path.decision_zone: {trigger: approach,
distance_m: 30}`).

**Kind** `intent_provider`, **factory**
`(intent_cfg, planner, *, marl_manager, training, previous)` returning a
callable `spawn context → RouteIntentDecision | None`. (Mechanics on
[Replace a Component](replace_component.md).)

## Worked example

The public helpers in `opencda_marl.api.route_intent` do the heavy
lifting — legal candidates from the planner's route table, the
feasibility mask, and the decision record the rest of the lifecycle
consumes:

```python
from opencda_marl.api import register_component
from opencda_marl.api.route_intent import (
    RouteIntent, build_candidates_with_diagnostics, build_decision,
    build_mask, resolve_route)


@register_component("intent_provider", "prefer_right")
def build_prefer_right(intent_cfg, planner, *, marl_manager=None,
                       training=True, previous=None):
    """Turn right whenever the junction allows it; otherwise keep."""
    def provider(context):
        event = context["event"]
        meta = event.metadata or {}
        if meta.get("entry_direction") is None:
            return None
        candidates, diagnostics = build_candidates_with_diagnostics(
            planner=planner,
            junction_id=int(event.junction_id),
            entry_direction=str(meta["entry_direction"]),
            entry_lane_idx=int(meta.get("entry_lane_idx", meta.get("lane_id", -1))),
            original_route_id=int(event.route_id),
        )
        if not candidates:
            return None
        mask = build_mask(candidates)
        intent = (RouteIntent.RIGHT if mask.get(RouteIntent.RIGHT)
                  else RouteIntent.KEEP)
        candidate = resolve_route(intent, candidates, mask,
                                  original_route_id=int(event.route_id))
        return build_decision(requested_intent=intent, candidate=candidate,
                              mask=mask,
                              original_route_id=int(event.route_id),
                              diagnostics=diagnostics)
    return provider
```

Selection runs through the action space: `MARL.action_space.path.provider`
names any registered `intent_provider`, next to the built-ins:

```yaml
MARL:
  action_space:
    mode: path_speed
    path:
      selector: rule_based     # a built-in selector: no learned head
      provider: prefer_right
```

Named next to a built-in selector (`rule_based`, `random_legal`,
`keep_current`), the provider decides alone and no learned head is built.
With `selector: learned`, or with `provider` named alone, it drives the
learned-head lifecycle and owns the head.

The `previous` argument is the provider being replaced on a world
reload; state that should outlive the rebuild (the built-in random
provider carries its RNG this way) is handed over through it.

## Verify

Copy `sumo-marl/intersection/route_rule_path_pcmappo.yaml` to
`my_configs/sumo-marl/intersection/prefer_right.yaml`, add
`MARL.action_space.path.provider: prefer_right` and
`MARL.plugins: [<the module holding build_prefer_right>]`, run it, and
read the decision counters — every spawn carries a `route_intent` decision
record in its metadata, and the monitor's episode summaries count them:

```bash
pixi run python opencda.py -t sumo-marl/intersection/prefer_right --marl --marl-config-root my_configs
```
