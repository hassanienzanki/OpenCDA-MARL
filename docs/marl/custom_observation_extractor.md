# An Observation Extractor

The extractor is what the learner sees: every step it turns the raw
per-agent observation dicts into the state vectors the policy consumes.
The policy manager constructs one per run; the declared feature set
(`MARL.<algo>.features`) decides the vector layout, and the extractor
decides how raw values become features.

**Kind** `observation_extractor`, **constructor**
`(config, algorithm_name, *, multi_agent)`, one method that matters:
`extract(observations) → {agent: np.ndarray}`. **Selected by**
`MARL.components.observation_extractor`. (Mechanics on
[Replace a Component](replace_component.md).)

## Worked example

Subclass the default and transform its output — a robustness check in
six lines:

```python
import numpy as np

from opencda_marl.api import register_component
from opencda_marl.builtin.extractors import ObservationExtractor


@register_component("observation_extractor", "clipped")
class ClippedExtractor(ObservationExtractor):
    """Clips every normalised feature, as a saturated sensor would."""

    def extract(self, observations):
        states = super().extract(observations)
        return {agent: np.clip(vec, -5.0, 5.0) for agent, vec in states.items()}
```

```yaml
MARL:
  components:
    observation_extractor: clipped
```

## The one contract to respect

The state dimension is part of the learner's contract: transform
features freely, but change the *width* only together with the learner —
or declare the feature set differently via `MARL.<algo>.features`, which
the default extractor already handles (that is also how the
`committed_exit_lanes` channel is switched on and sized).

## Verify

A width mismatch fails loudly at network construction; a value bug does
not. Run a short scenario and compare a logged state vector against the
raw observation it came from before trusting a training curve:

```bash
pixi run python opencda.py -t sumo-marl/intersection/route_speed_only_pcmappo --marl
```
