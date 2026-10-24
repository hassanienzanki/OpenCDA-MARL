# Derived maps

SUMO networks derived from OpenDRIVE, so a scenario can run on the **same
road network** on both backends instead of a CARLA map and a hand-built SUMO
lookalike. Everything here is generated — nothing is authored by hand:

```bash
pixi run python scripts/convert_opendrive_maps.py            # all sources
pixi run python scripts/convert_opendrive_maps.py --only Town10HD
```

## What is here

| file | source |
|---|---|
| `intersection.net.xml` | `opencda_marl/assets/intersection/intersection.xodr` (this repository's own map) |
| `Town*.net.xml`, `OpenDriveMap.net.xml` | the matching `.xodr` shipped with CARLA 0.9.16 |
| `IHP_map.net.xml`, `2lane_freeway_simplified.net.xml` | the vendored OpenCDA maps |
| `Town10HD.xodr` | copied from CARLA, so the building-free world below is self-contained |
| `conversion_report.json` | per map: netconvert result, junction/edge/lane/connection counts, whether SUMO loads it |

Generated with netconvert 1.27.1 (the `eclipse-sumo` package in this
environment) from CARLA 0.9.16. The CARLA-derived files are redistributions
of CARLA's map data; attribution belongs to the CARLA project.

## Which direction preserves identity

`.xodr → .net.xml` **keeps CARLA's junction ids** as SUMO node ids. That is
what makes a site chosen in CARLA findable in SUMO: a junction id read off
the CARLA map names the same node in the derived `.net.xml`. The reverse
(`--emit-opendrive`) renumbers junctions, so use it only to carry a
SUMO-side *edit* back to CARLA, never where identity matters.

## A town without buildings

CARLA can build a world from OpenDRIVE alone — road surface, markings and
signs, no buildings or props. Point a scenario at the `.xodr`:

```yaml
scenario:
  map:
    xodr_path: "opencda_marl/assets/maps/Town10HD.xodr"
```

Measured on CARLA 0.9.16: Town10HD generates in ~6 s with 200 spawn points,
and its junctions keep their ids — the same ids the SUMO network uses.
Because both sides come from one `.xodr`, the two backends describe the same
roads by construction.

The roads match; the frame takes one more input. A CARLA town export
mirrors CARLA's y axis, and SUMO reads a derived town network in CARLA's
frame only with junction landmarks: `runtime.coordinate_landmarks`, or a
`<net>_landmarks.json` (or `<net>_junction_graph.json`) beside the
`.net.xml`, keyed by junction id. None ship for the towns here, so without
them SUMO positions come out mirrored in y relative to CARLA's frame. SUMO
also does not read `scenario.map`: it loads the `<net-file>` of the
`.sumocfg` at `runtime.sumo_cfg`, and no `.sumocfg` ships for these
networks. The "Maps on both backends" section of the Run Experiments guide
covers both.
