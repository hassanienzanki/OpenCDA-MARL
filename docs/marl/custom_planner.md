# A Planner

The planner owns the route table: it finds the junctions, groups their
entry lanes, and generates the candidate routes traffic can drive. The
traffic manager constructs it once per scenario and queries it for every
spawn — on both backends — so replacing it changes what demand *is*.

**Kind** `planner`, **constructor** `(world, config)`, **selected by**
`scenario.traffic.planner.type`. (Mechanics — discovery, freeze,
override, validate — on [Replace a Component](replace_component.md).)

## Worked example

The built-in `marl` planner exposes its route queries as methods, so the
practical route is subclassing it and changing exactly the behaviour
you want to vary:

```python
from opencda_marl.api import register_component
from opencda_marl.builtin.planners.marl_planner import MARLPlanner


@register_component("planner", "keep_right")
class KeepRightPlanner(MARLPlanner):
    """Demand routes prefer a right turn wherever the junction has one."""

    def get_random_route(self, junction_id, visualize=False):
        right = self.get_random_route_by_type(junction_id, "right", visualize)
        return right if right is not None else super().get_random_route(
            junction_id, visualize)
```

```yaml
scenario:
  traffic:
    planner:
      type: keep_right
```

A from-scratch planner implements the same query surface the built-in
has (`get_junctions`, `get_routes`, `get_random_route*`); start from
`opencda_marl/builtin/planners/marl_planner.py` as the reference.

## Verify

Run the SUMO smoke and watch the *Traffic view* in the monitor — the
routes actually driven are the planner's output:

```bash
pixi run python opencda.py -t sumo-marl/intersection/route_rule_path_pcmappo --marl
```
