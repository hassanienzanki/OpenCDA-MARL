"""Manifest-based cell-level launcher for the experiment matrix.

This script coordinates independent matrix cells across machines without
introducing intra-run vectorized rollouts. Run ``--dry-run`` once to inspect the
global assignment, then run the same command on each machine with
``--worker <name>``. Each worker launches its assigned cells sequentially using
its configured CARLA ports and GPU id.
"""
from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from omegaconf import OmegaConf

from matrix_common import DEFAULT_MAP, DEFAULT_SLUG_PREFIX, canonical_maps, slug_for


REPO = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Worker:
    name: str
    os_name: str
    gpu_id: str
    carla_rpc_port: int
    carla_streaming_port: int
    max_concurrent_cells: int
    artifact_root: Path
    carla_root: str | None = None
    carla_bin: str | None = None


@dataclass(frozen=True)
class Cell:
    map_slug: str
    condition: str
    regime: str
    seed: int

    def slug(self, prefix: str) -> str:
        return slug_for(
            map_slug=self.map_slug,
            condition=self.condition,
            regime=self.regime,
            seed=self.seed,
            prefix=prefix,
        )


def _load_workers(path: Path) -> list[Worker]:
    data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    raw_workers = data.get("workers", []) if isinstance(data, dict) else []
    workers: list[Worker] = []
    for raw in raw_workers:
        worker = dict(raw)
        workers.append(
            Worker(
                name=str(worker["name"]),
                os_name=str(worker.get("os", "")),
                gpu_id=str(worker.get("gpu_id", 0)),
                carla_rpc_port=int(worker.get("carla_rpc_port", 2000)),
                carla_streaming_port=int(worker.get("carla_streaming_port", 0)),
                max_concurrent_cells=int(worker.get("max_concurrent_cells", 1)),
                artifact_root=Path(worker.get("artifact_root") or REPO),
                carla_root=worker.get("carla_root"),
                carla_bin=worker.get("carla_bin"),
            )
        )
    if not workers:
        raise ValueError(f"manifest has no workers: {path}")
    return workers


def _cells(maps: Iterable[str], conditions: Iterable[str],
           regimes: Iterable[str], seeds: Iterable[int]) -> list[Cell]:
    return [
        Cell(str(m), str(c), str(r), int(s))
        for m in maps for c in conditions for r in regimes for s in seeds
    ]


def _assign(cells: list[Cell], workers: list[Worker]) -> dict[str, list[Cell]]:
    slots: list[Worker] = []
    for worker in workers:
        if worker.max_concurrent_cells < 1:
            raise ValueError(f"worker {worker.name} has max_concurrent_cells < 1")
        slots.extend([worker] * worker.max_concurrent_cells)
    assignments = {worker.name: [] for worker in workers}
    for idx, cell in enumerate(cells):
        worker = slots[idx % len(slots)]
        assignments[worker.name].append(cell)
    return assignments


def _launch_worker_cells(
    worker: Worker,
    cells: list[Cell],
    args,
) -> int:
    if worker.max_concurrent_cells != 1:
        print(
            f"[dist] worker {worker.name} has max_concurrent_cells="
            f"{worker.max_concurrent_cells}; this launcher still starts cells "
            "sequentially in one process. Define one worker entry per GPU for "
            "true simultaneous execution.",
            flush=True,
        )
    script = worker.artifact_root / "scripts" / "experiments" / "launch_matrix.py"
    for cell in cells:
        cmd = [
            "pixi", "run", "python", str(script),
            "--maps", cell.map_slug,
            "--conditions", cell.condition,
            "--regimes", cell.regime,
            "--seeds", str(cell.seed),
            "--slug-prefix", args.slug_prefix,
            "--carla-rpc-port", str(worker.carla_rpc_port),
            "--carla-streaming-port", str(worker.carla_streaming_port),
            "--carla-restart-mode", args.carla_restart_mode,
            "--cuda-visible-devices", worker.gpu_id,
        ]
        if args.resume:
            cmd.append("--resume")
        if args.max_episodes is not None:
            cmd += ["--max-episodes", str(args.max_episodes)]
        if args.max_steps is not None:
            cmd += ["--max-steps", str(args.max_steps)]
        if args.cell_timeout is not None:
            cmd += ["--cell-timeout", str(args.cell_timeout)]
        if args.active_junctions is not None:
            cmd += ["--active-junctions"] + [str(j) for j in args.active_junctions]
        if getattr(args, "continue_source_prefix", None) is not None:
            cmd += ["--continue-source-prefix", args.continue_source_prefix]
        if worker.carla_root:
            cmd += ["--carla-root", str(worker.carla_root)]
        if worker.carla_bin:
            cmd += ["--carla-bin", str(worker.carla_bin)]
        print(f"[dist] {worker.name}: launching {cell.slug(args.slug_prefix)}",
              flush=True)
        print(f"[dist]   cmd: {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        rc = subprocess.run(cmd, cwd=worker.artifact_root).returncode
        if rc != 0:
            print(f"[dist] {cell.slug(args.slug_prefix)} exited rc={rc}",
                  flush=True)
            return rc
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="scripts/experiments/workers.example.yaml")
    p.add_argument("--worker", default=None,
                    help="Run only the assignments for this worker name.")
    p.add_argument("--maps", nargs="+", default=[DEFAULT_MAP])
    p.add_argument("--conditions", nargs="+",
                    default=["speed_only_pcmappo", "rule_path_pcmappo", "learn_path_pcmappo"])
    p.add_argument("--regimes", nargs="+", default=["aligned", "conflicting"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--cell-timeout", type=float, default=None)
    p.add_argument("--active-junctions", nargs="+", type=int, default=None,
                    help="Forwarded to launch_matrix.py: override "
                         "scenario.traffic.active_junctions for every map "
                         "(default: each map profile's own).")
    p.add_argument("--slug-prefix", default=DEFAULT_SLUG_PREFIX)
    p.add_argument("--continue-source-prefix", default=None,
                    help="Forwarded to launch_matrix.py: slug prefix of the "
                         "source cells that continue_* conditions start "
                         "from (default there: the --slug-prefix in effect).")
    p.add_argument("--carla-restart-mode", choices=["process", "none"],
                    default="process",
                    help="Forwarded to launch_matrix.py. Use none for CARLA "
                         "servers managed by Docker or another supervisor.")
    args = p.parse_args()

    # The canonical profile key names the planned cells and is what each
    # worker's launch_matrix.py receives.
    args.maps = canonical_maps(args.maps)
    workers = _load_workers(Path(args.manifest))
    cells = _cells(args.maps, args.conditions, args.regimes, args.seeds)
    assignments = _assign(cells, workers)

    print(f"[dist] planned cells: {len(cells)}", flush=True)
    for worker in workers:
        assigned = assignments[worker.name]
        print(
            f"[dist] worker={worker.name} os={worker.os_name} gpu={worker.gpu_id} "
            f"ports={worker.carla_rpc_port}/{worker.carla_streaming_port} "
            f"cells={len(assigned)} root={worker.artifact_root}",
            flush=True,
        )
        for cell in assigned:
            print(f"[dist]   {cell.slug(args.slug_prefix)}", flush=True)

    if args.worker is None:
        if args.dry_run:
            return
        raise SystemExit("Pass --worker <name> to run assigned cells on this machine.")

    selected = {worker.name: worker for worker in workers}.get(args.worker)
    if selected is None:
        raise SystemExit(f"unknown worker {args.worker!r}")
    raise SystemExit(_launch_worker_cells(selected, assignments[selected.name], args))


if __name__ == "__main__":
    main()
