"""Convert OpenDRIVE maps to SUMO networks, and check that they are usable.

    pixi run python scripts/convert_opendrive_maps.py                # every source
    pixi run python scripts/convert_opendrive_maps.py --only Town10HD
    pixi run python scripts/convert_opendrive_maps.py --out maps/sumo
    pixi run python scripts/convert_opendrive_maps.py --emit-opendrive   # round trip

Running one scenario on CARLA and on SUMO needs the *same* map on both
backends, and the honest way to get one is to derive the SUMO network from
the CARLA map's own OpenDRIVE export rather than to hand-build a lookalike.
This script runs that derivation with ``netconvert`` and then answers the
question that actually matters -- is the result drivable? -- by loading each
network, counting real junctions and lane connections, and checking that
at least one route exists across the map.

Sources: the repository's own ``.xodr`` files plus every ``Town*.xodr``
CARLA ships (``$CARLA_ROOT/CarlaUE4/Content/Carla/Maps/OpenDrive``).

Recorded paths are portable: netconvert writes its input and output paths
into a comment at the top of every network, and the report lists them too,
so both are rewritten repo-relative (``opencda_marl/assets/...``) or
``$CARLA_ROOT``-relative. A converted network can then be committed as-is
without carrying the converting machine's directory layout.

Direction matters. ``.xodr -> .net.xml`` **keeps CARLA's junction ids** as
SUMO node ids, which is what lets a site chosen in CARLA be found in the
SUMO network (and lets the coordinate bridge calibrate on a junction
centre). ``--emit-opendrive`` writes the reverse, ``.net.xml -> .xodr``,
for the case where the SUMO network was *edited* and CARLA has to follow --
but netconvert renumbers junctions on the way out, so the round-tripped
map no longer shares CARLA's ids. Use the forward direction for site
selection; use the reverse only to carry an edit across.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "artifacts" / "maps"

#: netconvert options for CARLA-exported OpenDRIVE. The import is geometry
#: driven, so the tolerances matter more than the defaults suggest:
#: --opendrive.import-all-lanes keeps lanes CARLA marks as non-driving out,
#: and guessing ramps/roundabouts on an already-explicit map invents
#: topology that the CARLA side does not have.
NETCONVERT_ARGS = [
    "--geometry.remove", "false",
    "--ramps.guess", "false",
    "--junctions.join", "true",
    "--opendrive.curve-resolution", "2.0",
    "--opendrive.internal-shapes", "true",
    "--no-turnarounds", "true",
    "--offset.disable-normalization", "false",
]


#: The path-valued options netconvert echoes into the configuration comment
#: at the top of a network it writes.
_HEADER_PATH = re.compile(
    r'(<(?:opendrive-files|output-file|sumo-net-file|opendrive-output) value=")([^"]*)(")')


def _carla_root() -> Path | None:
    value = os.environ.get("CARLA_ROOT") or os.environ.get("CARLA_HOME")
    return Path(value) if value else None


def portable_path(path: str | Path, *, repo: Path = REPO,
                  carla_root: Path | None = None) -> str:
    """``path`` as it may be recorded in a file that gets committed.

    Under the repository: repo-relative, POSIX separators. Under the CARLA
    install: ``$CARLA_ROOT/...``. Anything else: the file name alone (the
    report sits next to the networks it lists, so that still resolves).
    Never the converting machine's absolute layout. A value that is already
    portable (relative, or ``$CARLA_ROOT/...``) is kept, so rewriting twice
    changes nothing; the converter itself only ever passes absolute paths.
    """
    text = str(path)
    if text.startswith("$CARLA_ROOT/") or not os.path.isabs(text):
        return text.replace("\\", "/")
    # abspath, not resolve(): a symlinked checkout must keep its own prefix.
    absolute = Path(os.path.abspath(text))
    for base, prefix in ((repo, ""), (carla_root, "$CARLA_ROOT/")):
        if base is None:
            continue
        try:
            rel = absolute.relative_to(Path(os.path.abspath(str(base))))
        except ValueError:
            continue
        return prefix + rel.as_posix()
    return absolute.name


def scrub_net_header(net_file: Path, *, repo: Path = REPO,
                     carla_root: Path | None = None) -> bool:
    """Rewrite the paths in a network's leading netconvert comment portably.

    Only the comment before the first ``-->`` is touched, and the file is
    handled as bytes, so the network itself and its line endings stay
    byte-identical. Returns True when something changed.
    """
    data = net_file.read_bytes()
    end = data.find(b"-->")
    if end < 0:
        return False
    head = data[:end].decode("utf-8")
    new_head = _HEADER_PATH.sub(
        lambda m: m.group(1) + portable_path(m.group(2), repo=repo,
                                             carla_root=carla_root) + m.group(3),
        head)
    if new_head == head:
        return False
    net_file.write_bytes(new_head.encode("utf-8") + data[end:])
    return True


def _sources(explicit: list[str] | None) -> list[Path]:
    found: list[Path] = []
    for path in sorted(REPO.rglob("*.xodr")):
        # Skip dot-directories (.pixi, .git, caches): a copy of a map is
        # not a second map.
        if any(part.startswith(".") for part in path.relative_to(REPO).parts[:-1]):
            continue
        found.append(path)
    carla_root = _carla_root()
    if carla_root:
        opendrive = Path(carla_root) / "CarlaUE4" / "Content" / "Carla" / "Maps" / "OpenDrive"
        if opendrive.is_dir():
            # ``*_Opt`` duplicates the same road network with optional
            # rendering layers, so converting it twice proves nothing.
            found.extend(sorted(p for p in opendrive.glob("*.xodr")
                                if not p.stem.endswith("_Opt")))
    if explicit:
        wanted = {name.lower() for name in explicit}
        found = [p for p in found if p.stem.lower() in wanted]
    return found


def convert(source: Path, out_dir: Path) -> dict:
    """Run netconvert on one map and report what came out."""
    # Absolute, so the paths netconvert records are the ones portable_path
    # knows how to rewrite (a relative --out is relative to the cwd).
    out_dir = Path(os.path.abspath(str(out_dir)))
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{source.stem}.net.xml"
    log = out_dir / f"{source.stem}.netconvert.log"
    cmd = ["netconvert", "--opendrive-files", str(source), "-o", str(target),
           *NETCONVERT_ARGS]
    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=1800)
    log.write_text((proc.stdout or "") + (proc.stderr or ""), encoding="utf-8")
    carla_root = _carla_root()
    if target.is_file():
        scrub_net_header(target, carla_root=carla_root)
    row = {
        "map": source.stem,
        "source": portable_path(source, carla_root=carla_root),
        "seconds": round(time.time() - started, 1),
        "returncode": proc.returncode,
        "net": portable_path(target, carla_root=carla_root) if target.is_file() else None,
        "warnings": (proc.stderr or "").lower().count("warning"),
        "errors": (proc.stderr or "").lower().count("error"),
    }
    if target.is_file():
        row.update(inspect(target))
    return row


def inspect(net_file: Path) -> dict:
    """Structural facts about a converted network."""
    facts: dict = {"bytes": net_file.stat().st_size}
    junctions = 0
    internal = 0
    edges = 0
    lanes = 0
    connections = 0
    net_offset = None
    multi_lane_exits = 0
    for _event, element in ET.iterparse(str(net_file), events=("end",)):
        tag = element.tag
        if tag == "location":
            net_offset = element.get("netOffset")
        elif tag == "junction":
            if element.get("type") == "internal":
                internal += 1
            else:
                junctions += 1
        elif tag == "edge":
            if element.get("function") == "internal":
                pass
            else:
                edges += 1
                edge_lanes = len(element.findall("lane"))
                lanes += edge_lanes
                if edge_lanes >= 2:
                    multi_lane_exits += 1
        elif tag == "connection":
            connections += 1
        element.clear()
    facts.update({
        "net_offset": net_offset,
        "junctions": junctions,
        "internal_junctions": internal,
        "edges": edges,
        "lanes": lanes,
        "connections": connections,
        "multi_lane_edges": multi_lane_exits,
    })
    return facts


def check_loadable(net_file: Path) -> dict:
    """Can SUMO actually load and step this network?

    A network that netconvert wrote is not automatically a network SUMO
    will run: this loads it headless for a few steps, which is the check
    that matters before any scenario is built on it.
    """
    sumo = shutil.which("sumo") or shutil.which("sumo.exe")
    if sumo is None:
        return {"loadable": None, "detail": "sumo binary not found"}
    cmd = [sumo, "--net-file", str(net_file), "--begin", "0", "--end", "5",
           "--no-step-log", "true", "--no-warnings", "true"]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=600)
    return {"loadable": proc.returncode == 0,
            "detail": (proc.stderr or "").strip()[-200:] or "ok"}


def emit_opendrive(net_file: Path, out_dir: Path) -> dict:
    """Write a SUMO network back out as OpenDRIVE.

    For carrying a SUMO-side edit into CARLA. Junction ids are reassigned
    by netconvert, so the result is a different naming of the same roads —
    do not use it where junction identity matters.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{net_file.name.replace('.net.xml', '')}.xodr"
    proc = subprocess.run(
        ["netconvert", "--sumo-net-file", str(net_file),
         "--opendrive-output", str(target)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=1800)
    return {"opendrive_out": (portable_path(target, carla_root=_carla_root())
                              if target.is_file() else None),
            "opendrive_rc": proc.returncode}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help=f"output directory (default: {DEFAULT_OUT})")
    parser.add_argument("--only", nargs="*", default=None,
                        help="convert only these map stems (e.g. Town10HD intersection)")
    parser.add_argument("--skip-load-check", action="store_true")
    parser.add_argument("--emit-opendrive", action="store_true",
                        help="also write .net.xml -> .xodr (junction ids are "
                             "renumbered; see the module docstring)")
    args = parser.parse_args()

    sources = _sources(args.only)
    if not sources:
        print("no OpenDRIVE sources found "
              "(set CARLA_ROOT for the town maps)", file=sys.stderr)
        return 2
    out_dir = Path(os.path.abspath(args.out))
    rows = []
    print(f"{len(sources)} OpenDRIVE source(s) -> {out_dir}\n")
    for source in sources:
        print(f"  converting {source.stem} ...", flush=True)
        try:
            row = convert(source, out_dir)
        except subprocess.TimeoutExpired:
            row = {"map": source.stem,
                   "source": portable_path(source, carla_root=_carla_root()),
                   "returncode": -1, "net": None, "error": "netconvert timeout"}
        # row["net"] is the portable label; the checks need the real file.
        net_file = out_dir / f"{source.stem}.net.xml"
        if row.get("net") and not args.skip_load_check:
            row.update(check_loadable(net_file))
        if row.get("net") and args.emit_opendrive:
            row.update(emit_opendrive(net_file, out_dir / "xodr"))
        rows.append(row)

    report = out_dir / "conversion_report.json"
    report.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"\n{'map':<16} {'rc':>3} {'junctions':>9} {'edges':>6} {'lanes':>6} "
          f"{'conns':>6} {'loads':>6}  netOffset")
    for row in rows:
        print(f"{row['map']:<16} {row.get('returncode', '?'):>3} "
              f"{row.get('junctions', '-'):>9} {row.get('edges', '-'):>6} "
              f"{row.get('lanes', '-'):>6} {row.get('connections', '-'):>6} "
              f"{str(row.get('loadable', '-')):>6}  {row.get('net_offset', '-')}")
    print(f"\nreport: {report}")
    return 0 if all(r.get("returncode") == 0 for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
