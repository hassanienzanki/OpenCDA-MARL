"""The library's command line: run scenarios and list algorithms without the
repository checkout.

Installed as the ``opencda-marl`` console script by the wheel::

    opencda-marl list                                    # registered algorithms
    opencda-marl run -t sumo-marl/intersection/random_speed
    opencda-marl run -t sumo-marl/intersection/my_profile --config-root ./configs

``run`` drives the full training lifecycle (MARLCoordinator) for the SUMO
backend, resolving configs through the normal chain — explicit roots, the
current directory's ``configs/``, a repository checkout, then the configs
bundled into the wheel. The CARLA backend needs the vendored ``opencda``
core and a running server, so it is available from a repository checkout
(``python opencda.py … --marl``) rather than from the bare wheel; ``run``
says so instead of failing deep inside an import.

The repository's ``opencda.py`` remains the entry point for checkouts (it
also carries the non-MARL OpenCDA scenarios); this module is the subset a
pip install can honour.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_config(scenario_ref: str, config_roots: list[str]):
    from omegaconf import OmegaConf

    from opencda_marl.utils import resolve_marl_config_paths, smart_merge
    from opencda_marl.utils.config_locator import load_profile_with_includes
    from opencda_marl.utils.config_merger import normalize_run_config

    root = Path.cwd()
    resolved = resolve_marl_config_paths(root, scenario_ref,
                                         custom_config_roots=config_roots or None)
    merged = smart_merge(OmegaConf.load(resolved["default_yaml"]),
                         OmegaConf.load(resolved["scenario_base_yaml"]))
    merged = smart_merge(merged, load_profile_with_includes(resolved["scenario_yaml"]))
    # _cmd_run reads runtime.backend before the coordinator normalises, so a
    # section left empty is a mapping from here on.
    return normalize_run_config(merged)


def _cmd_run(args: argparse.Namespace) -> int:
    from loguru import logger

    # Preload torch before anything numpy-heavy, mirroring opencda.py: the
    # local Windows torch build faults when numpy's DLLs win the load order.
    # A slim install without torch is fine — the baselines are lazy and a
    # torch-free policy (e.g. the contrib example) never needs it.
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    except OSError as exc:  # pragma: no cover - machine-specific DLL state
        print(f"warning: torch failed to preload ({exc}); torch-based "
              "learners will not work in this process", file=sys.stderr)

    config = _build_config(args.test_scenario, args.config_root or [])
    backend = str(config.get("runtime", {}).get("backend", "carla")).lower()
    if backend != "sumo":
        print(f"Scenario {args.test_scenario!r} uses the {backend!r} backend. "
              "The bare wheel drives SUMO only; CARLA scenarios need the "
              "repository checkout (python opencda.py -t <ref> --marl) and a "
              "running CARLA server.", file=sys.stderr)
        return 2

    logger.remove()
    logger.add(sys.stdout, level="DEBUG" if args.debug else "INFO")

    # Mirror opencda.py's contract: the coordinator reads opt flags from the
    # config; the CLI is headless (no GUI) by definition.
    from omegaconf import OmegaConf

    config = OmegaConf.merge(config, OmegaConf.create(
        {"opt": {"test_scenario": args.test_scenario, "marl": True, "gui": False,
                 "debug": bool(args.debug), "apply_ml": False,
                 "marl_config_root": list(args.config_root or [])}}))

    from opencda_marl import MARLCoordinator

    coordinator = MARLCoordinator(config=config)
    try:
        coordinator.initialize()
        coordinator.run()
    finally:
        coordinator.close()
    return 0


def _cmd_list(_args: argparse.Namespace) -> int:
    from opencda_marl.api import algorithms as registry
    from opencda_marl.plugins import describe_plugins

    rc = registry._main()
    rows = [r for r in describe_plugins(load=False) if r["kind"] != "algorithm"]
    if rows:
        print(f"\n{'kind':20s} {'key':28s} source")
        for r in rows:
            print(f"{r['kind']:20s} {r['key']:28s} {r['source']}")
    return rc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="opencda-marl",
        description="OpenCDA-MARL: cooperative multi-agent RL for autonomous driving.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a scenario through the training lifecycle (SUMO)")
    run.add_argument("-t", "--test-scenario", required=True,
                     help="scenario reference, e.g. sumo-marl/intersection/random_speed")
    run.add_argument("--config-root", action="append", default=None,
                     help="extra config root (repeatable); the wheel's bundled "
                          "configs are always the last fallback")
    run.add_argument("--debug", action="store_true")
    run.set_defaults(func=_cmd_run)

    lst = sub.add_parser("list", help="list registered algorithms and their sources")
    lst.set_defaults(func=_cmd_list)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
