# PCMA: Preference-Coordinated Learners

The preference-conditioned learners in OpenCDA-MARL — **PCMASAC**,
**PCMAPPO** and the conditioning-only **PCMATD3** — build on *Preference
Coordinated Multi-agent Policy Optimization* (PCMA):

> Pengxin Wang, Lihao Guo, Yi Xie, Bo Liu, Siyang Cao, Jingdi Chen.
> **Learning Coordinated Preference for Multi-Objective Multi-Agent
> Reinforcement Learning.** arXiv:2606.14693, 2026.
> <https://arxiv.org/abs/2606.14693>

The paper's setting is cooperative multi-objective MARL: a team optimises
several, possibly conflicting objectives, and the conflicts run across
agents as well as across objectives. PCMA gives every agent its own
preference over the objectives and lets the agents *coordinate* those
preferences — shifting them so that the team's trade-offs are
complementary rather than identical. The signal-free intersection in this
framework is the paper's traffic-control scenario.

These learners are the framework's variant, not a reproduction of the
paper's algorithm: preferences are assigned (layer 2 below) and shifted by
a deterministic head, with a vector critic and a separation term, and the
paper's Dirichlet preference planner is not implemented. Do not report a
result from them as a result for the paper's PCMA. This page states exactly
what the implementation does, so results can be reproduced and the learners
extended.

## Three layers

A PCMA run is configured in three places, each independent of the others.

### 1. Objectives — `MARL.objective_mode`

The reward pipeline computes per-agent reward *components* (progress,
success, step penalty, collision, …; `MARL.reward_profile` selects their
weights). An **objective profile** groups components into \(K\) objectives
and the environment emits a vector \(\mathbf r_i \in \mathbb R^K\) per agent
per step alongside the scalar reward:

| `objective_mode` | objectives | components |
|---|---|---|
| `cooperative_pref_v1` | `efficiency`, `safety` | progress + success + step_penalty; collision |
| `competitive_ordering_v1` | `efficiency`, `interaction_risk` | progress + success + step_penalty; collision |

`MARL.reward_mode` decides what the learner is trained on: `objective_vector`
hands \(\mathbf r_i\) to a learner that declares `vector_rewards` and runs a
vector critic; otherwise the learner receives the scalarised
\(\mathbf w_i^\top \mathbf r_i\). Profiles are defined in
`opencda_marl/api/objectives.py`.

### 2. Preferences — who holds which weight vector

Each agent \(i\) carries a raw preference \(\mathbf w_i\) on the simplex
(\(K = 2\) today). It is assigned once per vehicle, in this order:

1. **Per-vehicle at spawn** (`agents.preference`, used by the
   `momarl/` and `sumo-marl/` experiment profiles): the traffic manager
   samples a preference for every spawned vehicle. `regime: aligned` gives
   everyone \([0.5, 0.5]\); `regime: conflicting` alternates
   \([0.75, 0.25]\) / \([0.25, 0.75]\) by spawn parity, so safety- and
   efficiency-weighted vehicles meet at the intersection.
2. **By profile** (`MARL.preference`, used by the `pcma/` profiles): a
   named table assigned by entry direction — `balanced` (\([0.5,0.5]\)),
   `mild_asymmetric_split` (\([0.6,0.4]\) for north/east, \([0.4,0.6]\) for
   south/west), `asymmetric_split` (0.8/0.2) and `reversed_asymmetric_split`.

`injection: state_augmentation` appends \(\mathbf w_i\) to the agent's
state vector; `MARL.state_dim` therefore includes `preference.dim`. Every
learner sees the preference in its observation — that alone is *direct*
conditioning, and it is all PCMATD3 does.

### 3. Learners — `MARL.pcmasac` / `MARL.pcmappo`

Both learners share one preference mechanism, the shift head below on
assigned preferences; they differ in the base algorithm (SAC off-policy
with twin critics; PPO on-policy with GAE).

**Preference shift.** In `internalpref` modes the actor carries a *shift
head* \(h\) on top of its observation features
\(\phi_i = \phi(o_i, c_i)\) (\(c_i\) is the LSTM-encoded neighbour context):

$$
\boldsymbol\delta_i = \operatorname{clip}\big(\tanh h(\phi_i),\ \pm\beta\big),\qquad
\tilde{\mathbf w}_i = \operatorname{norm}(\mathbf w_i + \boldsymbol\delta_i),\qquad
\bar{\mathbf w}_i = \operatorname{norm}\big((1-m)\,\mathbf w_i + m\,\tilde{\mathbf w}_i\big)
$$

with clip \(\beta\) (`pref_shift_clip` / `pref_shift_max_abs`), mix
\(m\) (`pref_shift_mix`) and `norm` the projection back onto the simplex.
The **effective preference** \(\bar{\mathbf w}_i\) — not the raw one — is
what conditions the actor (`pref_encoder_fusion`, `concat` or `film` in
PCMAPPO; an encoder-fusion head in PCMASAC). The head is initialised at zero
shift, so training starts from plain direct conditioning.

**Vector critic.** With `critic_mode: vector` the critic outputs
\(\mathbf Q(s,a) \in \mathbb R^K\) (PCMASAC) or a per-objective value head
next to the scalar one (PCMAPPO, trained on per-objective returns with
weight `critic_vector_l2_coef`). PCMASAC's actor then maximises the
preference-scalarised value

$$
J_i = \mathbb E\big[\ \bar{\mathbf w}_i^\top \mathbf Q(s, a) - \alpha \log \pi(a \mid s)\ \big],
$$

where `pref_update_mode` selects the weights: `shifted_pref_actor`
(\(\bar{\mathbf w}_i\), gradients flow into the shift head — the
shift-aware update), `shifted_pref_actor_detached` (same weights, no gradient), or
`direct_pref_actor` (\(\mathbf w_i\)). PCMAPPO's policy gradient uses the
advantage of the raw-preference scalarised reward
\(\mathbf w_i^\top \mathbf r_i\); the shift acts through the actor's
conditioning and the regularisers below.

**Regularisers.** Two terms added to the actor loss:

- shift magnitude, `pref_shift_l2_coef` \(\cdot\ \|\boldsymbol\delta_i\|^2\),
  keeps the learned preference near the assigned one;
- preference separation (`pref_mode: internalpref_sep`), a hinge on the
  effective preferences of agents in *different* raw-preference groups,
  `pref_sep_coef` \(\cdot \sum_{i<j,\ g_i \ne g_j} \max(0,\ \text{margin} - \|\bar{\mathbf w}_i - \bar{\mathbf w}_j\|)\),
  which keeps the shifted preferences of the two groups from collapsing
  onto each other. `pref_sep_use_effective_pref: false` applies it to
  \(\tilde{\mathbf w}_i\) instead.

The preference-conditioned learners also use `reward_aggregation:
team_mean` (declared in their registration), so every agent is trained on
the team's mean reward — the cooperative objective of the paper's setting.

## Configuration

```yaml
MARL:
  algorithm: pcmasac                  # or pcmappo
  reward_profile: cooperative_intersection_v1
  objective_mode: cooperative_pref_v1
  reward_mode: objective_vector       # vector critic; omit for scalarised rewards
  state_dim: 11                       # 9 features + 2 preference entries
  preference:
    profile: mild_asymmetric_split
    assignment: entry_direction
    injection: state_augmentation
    dim: 2
  pcmasac:
    pref_mode: internalpref           # direct | internalpref | internalpref_sep
    critic_mode: vector               # scalar | vector
    pref_dim: 2
    pref_update_mode: shifted_pref_actor
    pref_shift_clip: 0.5
    pref_shift_mix: 0.5
    pref_shift_l2_coef: 0.1
    pref_sep_coef: 0.02               # internalpref_sep only
    pref_sep_margin: 0.25
```

| key | PCMASAC | PCMAPPO | meaning |
|---|---|---|---|
| `pref_mode` | ✓ | ✓ | `direct`, `internalpref`, `internalpref_sep` |
| `critic_mode` | ✓ | ✓ | `scalar` or `vector` (`internalpref_sep` + vector is rejected in PCMASAC) |
| `pref_dim`, `pref_hidden_dim` | ✓ | ✓ | preference width and encoder size |
| `pref_update_mode` | ✓ | — | which weights scalarise \(\mathbf Q\) in the actor loss |
| `pref_shift_clip` / `pref_shift_max_abs` | ✓ | ✓ | \(\beta\) |
| `pref_shift_mix` | ✓ | ✓ | \(m\) |
| `pref_shift_l2_coef` | ✓ | ✓ | shift penalty |
| `pref_sep_coef`, `pref_sep_margin`, `pref_sep_use_effective_pref` | ✓ | ✓ (no `use_effective` switch) | separation hinge |
| `actor_pref_conditioning`, `critic_pref_conditioning`, `actor_use_pref_shift_head` | — | ✓ | `pref_encoder_fusion` / `concat` / `film`; force the shift head on or off |
| `critic_vector_l2_coef` | — | ✓ | weight of the per-objective value loss |

Everything not listed (network sizes, learning rates, entropy tuning,
rollout length, value normalization) is the base learner's.

## Profiles that ship

| scenario | what it is |
|---|---|
| `pcma/intersection/pcmasac_directpref_vector`, `pcmappo_directpref_vector` | CARLA cooperative benchmark: direct preference, vector critic, `mild_asymmetric_split` — the CARLA exemplars of these learners. `masac` (scalar SAC) and `vanilla` (no learner) next to them are one-episode checks on replay traffic, not budget- or traffic-matched to the exemplars |
| `momarl/intersection/{speed_only,keep_current_path,random_legal_path,rule_path,learn_path}_pcmappo` | the path + speed example family: per-vehicle preferences, aligned / conflicting regimes, speed-only or speed + path action space |
| `sumo-marl/intersection/route_{speed_only,rule_path,learn_path}_pcmappo` | the SUMO subset |

## What to watch

In the [monitor](../run-experiments.md#watch-it), beyond the usual losses:

- `SAC/pref_shift_l2`, `SAC/pref_shift_norm`, `SAC/pref_shift_abs_max`
  (PCMASAC) and `Pref/shift_l2` (PCMAPPO): how far the learned preferences
  moved. Zero throughout under `internalpref` or `internalpref_sep` means
  the shift head never trained. The `SAC/pref_shift_*` tags are logged only
  under those two modes, so if they are missing, `pref_mode` is `direct`
  (or the learner is plain MASAC).
- `SAC/effective_pref_obj_0_mean` vs `SAC/raw_pref_obj_0_mean` (vector
  critic, internal modes): the direction of the shift.
- `SAC/pref_sep_loss`, `SAC/pref_sep_active_pairs` (PCMASAC, logged only
  under `internalpref_sep`) / `Pref/sep_loss` (PCMAPPO): the separation term
  and how many cross-group pairs were in the batch.
- `Q_values/Q1_obj_k_mean` (PCMASAC, vector critic) and `Loss/vector_value`
  (PCMAPPO): per-objective value estimates.

Each episode's competition summary (in the metrics archive and the
evaluation output) splits success, collision and timeout rates and the
average completion rank by preference group, so the two groups can be
compared directly.

## Citation

If you use the preference-conditioned learners, please cite the PCMA paper
together with the framework paper ([README](https://github.com/radar-lab/OpenCDA-MARL#citation)):

```bibtex
@misc{wang2026pcma,
  title={Learning Coordinated Preference for Multi-Objective Multi-Agent Reinforcement Learning},
  author={Pengxin Wang and Lihao Guo and Yi Xie and Bo Liu and Siyang Cao and Jingdi Chen},
  year={2026},
  eprint={2606.14693},
  archivePrefix={arXiv},
  primaryClass={cs.MA},
  url={https://arxiv.org/abs/2606.14693},
}
```
