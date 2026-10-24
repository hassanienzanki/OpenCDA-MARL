# Replace a Component

Every joint in the lifecycle swaps the same way. This page is the
mechanism; each kind then has its own page with a complete worked
example:

| kind | what it decides | page |
|---|---|---|
| `algorithm` | how agents learn and act | [Add an Algorithm](custom_algorithms.md) |
| `planner` | the route table the traffic drives | [A Planner](custom_planner.md) |
| `agent` | the driver behind one vehicle | [An Agent Type](custom_agent.md) |
| `intent_provider` | which way a vehicle goes at the junction | [An Intent Provider](custom_intent_provider.md) |
| `observation_extractor` | what the learner sees | [An Observation Extractor](custom_observation_extractor.md) |
| `reward_calculator` | the structure of the reward | [A Reward Calculator](custom_reward_calculator.md) |
| `reward_profile` / `objective_profile` / `preference_profile` | the numbers, as named tables | [Profile Tables](custom_profiles.md) |

The full map — every joint's contract, door, config key and built-ins in
one table — is in [Architecture](architecture.md); this section is the
practice.

## How a component reaches the framework

Your component lives in *your* package (or, on a checkout, a module
dropped into `opencda_marl/builtin/algorithms/contrib/`). It reaches the
registry through the same doors an external wheel uses — an
`opencda_marl.plugins` entry point, `MARL.plugins` in config, or
`$OPENCDA_MARL_PLUGINS`
([discovery](custom_algorithms.md#making-it-discoverable)) — by calling a
`register_*` function at import time:

```python
from opencda_marl.api import register_component

@register_component("planner", "my_planner")
class MyPlanner:
    ...
```

At startup the framework loads every plugin source once, then **freezes
the registry** — no hot reload, so what ran is exactly what
`pixi run python -m opencda_marl.plugins` lists. A plugin whose declared
API version mismatches is rolled back as a unit, across every kind it
registered.

## How the framework picks yours

Selection is configuration, never code: each kind has a config key
(`scenario.traffic.planner.type`, `agents.agent_type`,
`MARL.components.<kind>`, `MARL.reward_profile`, …— the
[Architecture](architecture.md#replaceable-joints) table maps all of
them). The lifecycle resolves the key through the registry at
initialize, so swapping implementations is a one-line profile change.

## The rules

- **Signatures are documented in one place** — `KIND_SIGNATURES` in
  `opencda_marl.api.components` states every kind's constructor shape.
- **Replacing an existing key** — built-ins included — **requires
  `override=True`**, so nothing is shadowed by a typo.
- **`register_component(..., validate=fn)`** attaches your own config
  check: a callable over the selecting config block, raising
  `ValueError` on a bad configuration. It runs when the framework
  resolves the component, so a misconfigured plugin fails at initialize
  with your message instead of deep in the loop.

## An axis of your own

The registry accepts unknown kinds — a new research axis must not need
an engine release. Register under your own kind, resolve it from your
own code, and it is listed with everything else:

```python
from opencda_marl.api import register_component
from opencda_marl.api.components import resolve_component


@register_component("curriculum", "linear_demand")
class LinearDemandCurriculum:
    ...

curriculum = resolve_component("curriculum", "linear_demand")()   # your code
```

## Verifying a component

The suite's contract tests do not know your component exists, so the
check is behavioural, the same as for [an
algorithm](custom_algorithms.md#verifying-your-algorithm): run a short
scenario and read the counters the monitor (or the log) reports — spawns
carrying your decisions, the planner's routes actually driven, the
reward terms in the episode summaries. For anything that runs on both
backends (every kind but `agent`, which is resolved on CARLA only), the
SUMO smoke is the fastest loop:

```bash
pixi run python opencda.py -t sumo-marl/intersection/route_rule_path_pcmappo --marl
```
