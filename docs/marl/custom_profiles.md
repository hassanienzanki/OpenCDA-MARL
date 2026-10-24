# Profile Tables

Rewards, objectives and preferences are *data*, not code: named tables
selected by key. An experiment that only changes numbers — a reward weight, an
objective split, a preference assignment — ships a profile, keeps every
line of framework code identical, and stays diffable as configuration.

**Kinds** `reward_profile`, `objective_profile`, `preference_profile`;
**registered by** `register_reward_profile` /
`register_objective_profile` / `register_preference_profile`;
**selected by** `MARL.reward_profile`, `MARL.objective_mode`,
`MARL.preference.profile`. (Mechanics on
[Replace a Component](replace_component.md).)

## Worked example

```python
from opencda_marl.api import register_reward_profile

register_reward_profile("lab_rewards", {"rewards": {"success": 500.0}})
```

```yaml
MARL:
  reward_profile: lab_rewards
```

`register_objective_profile` and `register_preference_profile` work the
same way. The built-in tables are the reference shapes to copy from —
they live next to their register functions in
`opencda_marl/api/rewards.py`, `objectives.py` and `preferences.py`
(e.g. `competitive_intersection_v2`, `cooperative_pref_v1`, `balanced`).

## How the three relate

- A **reward profile** parameterises the
  [calculator's](custom_reward_calculator.md) terms.
- An **objective profile** defines the objective vector those terms are
  folded into.
- A **preference profile** assigns the weights over that vector — by
  entry direction here; the per-vehicle *sampler* is configuration
  instead (`agents.preference`), because sampled preferences are a
  distribution, not a table.

The [PCMA learners](pcma.md) are built on exactly this split.

## Verify

A typo'd key fails at initialize (unknown profiles do not resolve), and
overriding an existing name requires `override=True`. The record that
makes a summary attributable to its numbers is the run's own config: the
selecting keys live in the profile YAML that launched it, and matrix
cells persist their fully merged runtime profile next to the run.
