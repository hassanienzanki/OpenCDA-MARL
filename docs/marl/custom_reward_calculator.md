# A Reward Calculator

The calculator is the *structure* of the reward — which terms exist and
how they are computed. Changing the *numbers* is a
[reward profile](custom_profiles.md); a new calculator is for new terms
or new shaping. Each env constructs one at initialize and calls it every
step.

**Kind** `reward_calculator`, **constructor**
`(reward_params, reward_spec)`, **selected by**
`MARL.components.reward_calculator`. (Mechanics on
[Replace a Component](replace_component.md).)

## Worked example

The built-in exposes its terms as methods, and the term methods are
shared by both backends' entry points (`calculate_carla` /
`calculate_sumo`) — so one term can be reshaped in isolation and the
change holds on CARLA and SUMO alike:

```python
from opencda_marl.api import register_component
from opencda_marl.builtin.rewards.calculator import RewardCalculator


@register_component("reward_calculator", "impatient")
class ImpatientRewards(RewardCalculator):
    """Doubles the progress signal; every other term inherited."""

    def calculate_progress_reward(self, current_observation, previous_observation):
        return 2.0 * super().calculate_progress_reward(
            current_observation, previous_observation)
```

```yaml
MARL:
  components:
    reward_calculator: impatient
```

For multi-objective work the calculator's output feeds the objective
vector defined by `MARL.objective_mode`
([Profile Tables](custom_profiles.md)); the reward/objective split means
a new term composes with preference conditioning instead of fighting it.

## Verify

Reward changes are the easiest to fool yourself with: read the per-term
components the calculator emits (`RewardOutput.components`), not just
the total, and compare one episode against the unmodified calculator
before training on it:

```bash
pixi run python opencda.py -t sumo-marl/intersection/route_speed_only_pcmappo --marl
```
