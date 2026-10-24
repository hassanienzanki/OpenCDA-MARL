# An Agent Type

An agent type is the driver behind one vehicle: the factory is called
per spawned vehicle and returns whatever drives it — a learner's
actuator, a rule-based baseline, a scripted human model. The built-ins
(`marl`, `rule_based`, `vanilla`, `behavior`) register through the same
door yours does.

**Kind** `agent`, **factory** `(vehicle, carla_map, config)`,
**selected by** `agents.agent_type` — or per vehicle through the
population mix, which is where this kind earns its keep. (Mechanics on
[Replace a Component](replace_component.md).)

!!! note "CARLA only"
    The `agent` kind is resolved on the CARLA backend, one agent per
    spawned vehicle. On SUMO a population key is a label: vehicles outside
    the learner's population drive under SUMO's own car-following model,
    and a registered agent factory never runs. The no-learner baselines are
    exactly `behavior`, `vanilla` and `rule_based`; a registered agent type
    is never treated as one.

## Worked example

The built-in factories compose, so a new type can be a
re-parameterisation of an existing one — here, the behavior baseline
slowed down, reading its own config section:

```python
from opencda_marl.api import register_component
from opencda_marl.builtin.agents.agent_factory import build_behavior_agent


@register_component("agent", "cautious_driver")
def build_cautious_driver(vehicle, carla_map, config):
    """The behavior baseline, slowed down; reads agents.cautious_driver."""
    cfg = dict(config or {})
    cfg["behavior"] = {**(cfg.get("behavior") or {}),
                       "max_speed": 35,
                       **(cfg.get("cautious_driver") or {})}
    return build_behavior_agent(vehicle, carla_map, cfg)
```

Selected globally, or as a share of the population — a mixed-autonomy
experiment is just a distribution over registered agent types, with shares
exact over every window of `resolution` (default 100) vehicles and
reproducible per seed:

```yaml
agents:
  agent_type: cautious_driver          # everyone
  # or:
  population:
    mix: {marl: 0.7, cautious_driver: 0.3}
    seed: 7
```

The learner commands — and learns from — its own population only, so the
mixed run stays a clean experiment. On CARLA the mix is checked at
construction: it must name the learning population (`agents.agent_type`),
and every name with a positive share must be a registered agent type
(matched exactly, case-sensitive), so a typo fails before the first episode
rather than at its first spawn.

## Verify

On a CARLA profile: copy `momarl/intersection/speed_only_pcmappo.yaml` to
`my_configs/momarl/intersection/cautious_mix.yaml`, add
`agents.population: {mix: {marl: 0.7, cautious_driver: 0.3}, seed: 7}` and
`MARL.plugins: [<your module>]`, start a CARLA server, and run it:

```bash
pixi run python opencda.py -t momarl/intersection/cautious_mix --marl --marl-config-root my_configs
```

Then read the spawn log or the trace: each vehicle's agent type is in its
spawn metadata, and the shares hold exactly over every 100 consecutive
draws of the population pattern.
