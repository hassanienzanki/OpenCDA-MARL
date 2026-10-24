# Intersection Scenario Assets

This directory contains the shared intersection scenario assets for MARL runs.

**Location:** `opencda_marl/assets/intersection/`

The folder groups both CARLA-facing and SUMO-facing representations of the same
scenario so asset lookup is driven by scenario identity rather than simulator.

## Files

- `intersection.xodr` - OpenDRIVE map for CARLA local map loading
- `intersection.fbx` - source geometry asset used during map authoring/export
- `intersection.net.xml` - SUMO network converted from `intersection.xodr`
- `intersection_route_rich.con.xml` - extra lane connections, so every entry
  lane can reach every exit lane of its target road
- `intersection_route_rich.net.xml` - the SUMO network the `.sumocfg` loads:
  `intersection.net.xml` with the connections above applied
- `intersection.rou.xml` - Route definitions for intersection scenario
- `intersection.sumocfg` - SUMO configuration file

## Usage

These files are automatically loaded by the CARLA and SUMO scenario paths when
using the intersection scenario.

Configuration in `opencda_marl/_bundled/configs/sumo-marl/intersection/base.yaml`:
```yaml
runtime:
  backend: "sumo"
  sumo_cfg: "opencda_marl/assets/intersection/intersection.sumocfg"
```

## Manual Testing

To test the SUMO network manually:
```bash
sumo-gui -c intersection.sumocfg
```

## Regeneration

To regenerate the converted network from the XODR source (written under
`artifacts/maps/` unless `--out` says otherwise):
```bash
pixi run python scripts/convert_opendrive_maps.py --only intersection
```

`intersection_route_rich.net.xml` is then rebuilt with `netconvert`
(`--sumo-net-file intersection.net.xml --connection-files
intersection_route_rich.con.xml`); the full option set is recorded in the
comment at the top of the network file itself.
