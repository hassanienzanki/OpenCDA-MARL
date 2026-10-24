"""One-command verification battery for the lifecycle and the environment.

    pixi run verify              # fast: suite, registry, components, SUMO bindings + smoke
    pixi run verify --full       # + a CARLA episode (needs a running server)

Made for repeated runs: every check prints PASS/FAIL with its evidence
line, the battery keeps going after a failure, and the exit code is the
number of failed checks. Artifacts land under artifacts/verify/ (wiped per
run) so repeated verification never touches real run data.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VERIFY_ROOT = REPO / "artifacts" / "verify"
RESULTS: list[tuple[str, bool, str]] = []


def run(cmd: list[str], timeout: int = 1800) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=REPO, text=True, capture_output=True,
                          encoding="utf-8", errors="replace", timeout=timeout)


def check(name: str, ok: bool, evidence: str) -> None:
    RESULTS.append((name, ok, evidence))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {evidence}", flush=True)


def check_suite() -> None:
    t0 = time.time()
    proc = run([sys.executable, "-m", "pytest", "test", "-q", "-p", "no:cacheprovider"])
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr[-200:]
    check("pytest test", proc.returncode == 0, f"{tail} ({time.time()-t0:.0f}s)")


def check_registry() -> None:
    proc = run([sys.executable, "-m", "opencda_marl.api.algorithms"])
    keys = re.findall(r"^(\w+)\s+\S+\s+(?:per_agent|team_mean)", proc.stdout, re.M)
    check("registry", proc.returncode == 0 and len(keys) >= 8,
          f"{len(keys)} algorithms via entry points")


def check_components() -> None:
    code = ("from opencda_marl.plugins import describe_plugins;"
            "rows = describe_plugins();"
            "kinds = sorted({r['kind'] for r in rows});"
            "print(len(rows), kinds)")
    proc = run([sys.executable, "-c", code])
    needed = ("agent", "planner", "algorithm", "reward_profile",
              "objective_profile", "preference_profile")
    ok = proc.returncode == 0 and all(k in proc.stdout for k in needed)
    check("plugins", ok, proc.stdout.strip()[-160:] or proc.stderr.strip()[-120:])


#: Run in a child interpreter: resolve SUMO the way the engine does, with the
#: separately installed traci/sumolib/libsumo distributions hidden, so the
#: only bindings left are the ones inside the eclipse-sumo package -- what a
#: plain `pip install` of the wheel provides. SUMO_HOME is unset for the
#: child, and a foreign SUMO tools directory elsewhere on sys.path (e.g. from
#: PYTHONPATH) fails the check: the resolver must put the package's first.
BARE_SUMO_PROBE = r"""
import importlib.machinery, importlib.util, os, sys, sysconfig
from pathlib import Path

HIDDEN = ("traci", "sumolib", "libsumo")


def norm(path):
    return os.path.normcase(str(Path(path or ".").resolve()))


SITE = {norm(sysconfig.get_paths()[k]) for k in ("purelib", "platlib")}


class OnlyThePackagesBindings:
    def find_spec(self, name, path=None, target=None):
        if name not in HIDDEN:
            return None
        dirs = [p for p in sys.path if norm(p) not in SITE]
        spec = importlib.machinery.PathFinder.find_spec(name, dirs)
        if spec is None:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return spec


sys.meta_path.insert(0, OnlyThePackagesBindings())
from opencda_marl._kernel.traffic import _sumo_lib
import sumolib, traci

tools = Path(importlib.util.find_spec("sumo").origin).resolve().parent / "tools"
for module in (traci, sumolib):
    where = Path(module.__file__).resolve()
    assert tools in where.parents, f"{module.__name__} came from {where}, not {tools}"
print(f"binding={_sumo_lib.SUMO_BINDING} traci+sumolib from {tools}")
"""


def check_bare_sumo_bindings() -> None:
    """The wheel declares only eclipse-sumo for the SUMO backend: traci and
    sumolib must import from inside that package, with nothing else installed."""
    env = {k: v for k, v in os.environ.items() if k != "SUMO_HOME"}
    proc = subprocess.run([sys.executable, "-c", BARE_SUMO_PROBE], cwd=REPO, env=env,
                          text=True, capture_output=True, encoding="utf-8",
                          errors="replace", timeout=300)
    lines = (proc.stdout.strip() or proc.stderr.strip()).splitlines()
    check("sumo bindings from eclipse-sumo alone", proc.returncode == 0,
          lines[-1][-200:] if lines else f"rc={proc.returncode}, no output")


def sumo_profile(out_root: Path) -> str:
    """The SUMO smoke profile. The monitor is off: a battery must not leave
    a detached server running, holding a port and the verify root."""
    return f"""description: "verify battery: PCMAPPO lifecycle smoke"
world: {{seed: 5, sumo_gui: false}}
scenario:
  simulation: {{max_steps: 900, max_episodes: 2}}
  traffic:
    mode: "replay"
    replay_file: "recordings/lite_2min_SUMO.json"
    base_speed: 45.0
    active_junctions: [4]
outputs:
  root_dir: "{out_root.as_posix()}"
  evaluation: {{mode: "disabled"}}
  tensorboard: {{enabled: false}}
  monitor: {{enabled: false}}
  metrics_archive: {{enabled: true, export_interval: 1, dir: "metrics_history/verify"}}
MARL:
  algorithm: "pcmappo"
  reward_profile: "competitive_intersection_v2"
  objective_mode: "competitive_ordering_v1"
  state_dim: 11
  action_dim: 1
  action_space: {{mode: speed_only}}
  preference: {{profile: "balanced", assignment: "entry_direction", injection: "state_augmentation", dim: 2}}
  pcmappo:
    pref_mode: "internalpref_sep"
    pref_dim: 2
    n_steps: 128
    batch_size: 32
    warmup_steps: 0
    max_action: 45.0
    min_action: 0.0
    features: {{rel_x: 1, rel_y: 1, position_x: 1, position_y: 1, lane_position: 1,
               heading_angle: 1, dist_to_intersection: 1, dist_to_front_vehicle: 1,
               waypoint_buffer: 1}}
  training:
    training_mode: true
    save_checkpoints: true
    save_freq: 1
    checkpoint_dir: "{(out_root / 'checkpoints').as_posix()}"
"""


def check_sumo_lifecycle() -> None:
    """A real SUMO training run: learner counters must advance."""
    out_root = VERIFY_ROOT / "sumo"
    # A profile-only extra root: default and base resolve from the bundle.
    profile_dir = REPO / "artifacts" / "_runtime" / "sumo-marl" / "intersection"
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile = profile_dir / "verify_sumo_pcmappo.yaml"
    profile.write_text(sumo_profile(out_root), encoding="utf-8")
    proc = run([sys.executable, "opencda.py", "-t",
                "sumo-marl/intersection/verify_sumo_pcmappo", "--marl",
                "--marl-config-root", str(REPO / "artifacts" / "_runtime")])
    episodes = proc.stdout.count("Episode reset for pcmappo") + \
        proc.stderr.count("Episode reset for pcmappo")
    counters = ""
    try:
        sys.path.insert(0, str(REPO))
        from opencda_marl.telemetry import MetricsStore, discover_runs

        store = MetricsStore(discover_runs(out_root)[0].path, read_only=True)
        tags = {t["tag"]: t for t in store.tags()}
        steps = tags.get("PCMAPPO/steps_collected", {}).get("last_value", 0)
        updates = tags.get("PCMAPPO/updates_performed", {}).get("last_value", 0)
        store.close()
        counters = f"episodes={episodes} steps_collected={steps:.0f} updates={updates:.0f}"
        ok = episodes >= 2 and steps > 1000 and updates >= 1
    except Exception as exc:  # noqa: BLE001
        counters, ok = f"episodes={episodes}; store unreadable: {exc}", False
    latest = out_root / "checkpoints" / "pcmappo" / "latest_checkpoint.pth"
    check("sumo lifecycle (pcmappo)", ok and latest.is_file(),
          counters + (", checkpoint saved" if latest.is_file() else ", NO checkpoint"))


def carla_profile(out_root: Path) -> str:
    """The CARLA lifecycle profile: the bundled rule_based baseline with its
    outputs under the verify root and the monitor off, for the same reason
    as the SUMO smoke (the bundled profile's default ``outputs`` would start
    a detached monitor on the shared ``artifacts`` root and write there)."""
    import yaml

    sys.path.insert(0, str(REPO))
    from opencda_marl._bundled import BUNDLED_CONFIG_ROOT

    cfg = yaml.safe_load((BUNDLED_CONFIG_ROOT / "marl" / "intersection" / "rule_based.yaml")
                         .read_text(encoding="utf-8"))
    cfg["description"] = "verify battery: CARLA rule_based lifecycle"
    outputs = cfg.setdefault("outputs", {})
    outputs["root_dir"] = out_root.as_posix()
    outputs["monitor"] = {"enabled": False}
    outputs["tensorboard"] = {"enabled": False}
    return yaml.safe_dump(cfg, sort_keys=False)


def check_carla_lifecycle() -> None:
    """--full only: one CARLA rule_based episode against a running server."""
    probe = run([sys.executable, "-c",
                 "import carla; c=carla.Client('127.0.0.1',2000); c.set_timeout(5.0);"
                 "print(c.get_server_version())"], timeout=60)
    if probe.returncode != 0:
        check("carla lifecycle", False,
              "no CARLA server on :2000 - start one and rerun with --full")
        return
    profile_dir = REPO / "artifacts" / "_runtime" / "marl" / "intersection"
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "verify_carla_rule_based.yaml").write_text(
        carla_profile(VERIFY_ROOT / "carla"), encoding="utf-8")
    proc = run([sys.executable, "opencda.py", "-t", "marl/intersection/verify_carla_rule_based",
                "--marl", "--marl-config-root", str(REPO / "artifacts" / "_runtime")],
               timeout=1200)
    text = proc.stdout + proc.stderr
    m = re.search(r"Episode summary: (\{.*\})", text)
    ok = proc.returncode == 0 and m is not None
    if not ok:
        # Preserve the evidence: a transient failure with discarded output
        # cannot be diagnosed on the next round.
        fail_log = VERIFY_ROOT / "carla_fail.log"
        fail_log.parent.mkdir(parents=True, exist_ok=True)
        fail_log.write_text(text[-20000:], encoding="utf-8", errors="replace")
    check("carla lifecycle (rule_based)", ok,
          (m.group(1)[:120] + "...") if m
          else f"rc={proc.returncode}, no episode summary - tail in {VERIFY_ROOT / 'carla_fail.log'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help="also run a CARLA episode (server on :2000 required)")
    parser.add_argument("--skip-suite", action="store_true",
                        help="skip pytest (when it just ran)")
    args = parser.parse_args()

    shutil.rmtree(VERIFY_ROOT, ignore_errors=True)
    if not args.skip_suite:
        check_suite()
    check_registry()
    check_components()
    check_bare_sumo_bindings()
    check_sumo_lifecycle()
    if args.full:
        check_carla_lifecycle()

    failed = [name for name, ok, _ in RESULTS if not ok]
    print(f"\nverify: {len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed"
          + (f" - FAILED: {', '.join(failed)}" if failed else ""))
    return len(failed)


if __name__ == "__main__":
    raise SystemExit(main())
