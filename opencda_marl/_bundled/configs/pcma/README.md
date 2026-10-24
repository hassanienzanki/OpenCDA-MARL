# PCMA Config Family

This folder contains the CARLA preference-conditioned speed-control profiles (PCMASAC / PCMAPPO, plus one-episode MASAC and vanilla checks on replay traffic):

- `pcma/intersection/*` is the speed-control and preference-conditioned controller tier.
- `momarl/intersection/*` is the path + speed example family.

Examples:

```bash
pixi run python opencda.py \
    -t pcma/intersection/smoke \
    --marl

pixi run python opencda.py \
    -t pcma/intersection/pcmasac_directpref_vector \
    --marl --apply_ml
```
