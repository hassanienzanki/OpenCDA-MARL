# MOMARL path + speed example family

One learner (PCMAPPO) under every built-in path provider, on the CARLA
intersection:

- `momarl/intersection/*` pairs a one-shot path decision with a per-step speed head under one algorithm object: `speed_only` (no path action), `keep_current_path`, `random_legal_path`, `rule_path` (the built-in path providers) and `learn_path` (a learned path head).
- The shared `default.yaml` plus scenario `base.yaml` files keep these profiles self-contained.
- The preference-conditioned speed-control profiles live in `pcma/`; the SUMO subset is `sumo-marl/intersection/route_*`.

Example:

```bash
pixi run python opencda.py \
    -t momarl/intersection/learn_path_pcmappo \
    --marl
```
