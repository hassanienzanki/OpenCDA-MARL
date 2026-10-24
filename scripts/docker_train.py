"""Run OpenCDA-MARL training commands in a bind-mounted Linux container.

The training image is optional. It does not contain the repository; instead the
host checkout is mounted into /workspace/OpenCDA-MARL so edits are picked up
without rebuilding. CARLA should run separately, commonly via
scripts/docker_carla.py.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence


REPO = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = "opencda-marl-train:cuda128"
DEFAULT_DOCKERFILE = REPO / "dependencies" / "docker" / "Dockerfile.train"
DEFAULT_WORKDIR = "/workspace/OpenCDA-MARL"
HELPER_OPTIONS = {
    "--context",
    "--dockerfile",
    "--dry-run",
    "--gpu",
    "--image",
    "--network",
    "--no-tty",
    "--pixi-volume",
    "--repo",
    "--tty",
    "--user",
}


def _option_name(arg: str) -> str:
    return arg.split("=", 1)[0]


def _normalize_pixi_separator(argv: Sequence[str]) -> list[str]:
    args = list(argv)
    if len(args) >= 3 and args[1] == "--" and _option_name(args[2]) in HELPER_OPTIONS:
        return [args[0], *args[2:]]
    return args


def _strip_separator(args: Sequence[str]) -> list[str]:
    args = list(args)
    if args and args[0] == "--":
        return args[1:]
    return args


def _quote(cmd: Sequence[str]) -> str:
    return shlex.join(str(part) for part in cmd)


def _run(cmd: Sequence[str], *, dry_run: bool = False) -> int:
    print(f"[docker-train] $ {_quote(cmd)}", flush=True)
    if dry_run:
        return 0
    return int(subprocess.run(list(cmd), check=False).returncode)


def _gpu_args(gpu: str) -> list[str]:
    gpu = str(gpu).strip()
    if not gpu or gpu.lower() == "all":
        return ["--gpus", "all", "-e", "NVIDIA_VISIBLE_DEVICES=all"]
    return ["--gpus", f"device={gpu}", "-e", f"NVIDIA_VISIBLE_DEVICES={gpu}"]


def _base_run_cmd(args: argparse.Namespace, *, interactive: bool) -> list[str]:
    cmd = ["docker", "run", "--rm"]
    if interactive:
        cmd.append("-it")
    cmd += [
        "--network",
        args.network,
    ]
    cmd += _gpu_args(args.gpu)
    cmd += [
        "-e",
        "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
        "-e",
        "SDL_VIDEODRIVER=offscreen",
        "-e",
        "SUMO_HOME=/usr/share/sumo",
        "-v",
        f"{args.repo.resolve()}:{DEFAULT_WORKDIR}",
        "-v",
        f"{args.pixi_volume}:{DEFAULT_WORKDIR}/.pixi",
        "-w",
        DEFAULT_WORKDIR,
    ]
    if args.user:
        cmd += ["--user", args.user]
    return cmd


def cmd_build(args: argparse.Namespace) -> int:
    cmd = [
        "docker",
        "build",
        "-f",
        str(args.dockerfile),
        "-t",
        args.image,
        str(args.context),
    ]
    return _run(cmd, dry_run=args.dry_run)


def cmd_shell(args: argparse.Namespace) -> int:
    cmd = _base_run_cmd(args, interactive=not args.no_tty)
    cmd += [args.image, "/bin/bash"]
    return _run(cmd, dry_run=args.dry_run)


def cmd_run(args: argparse.Namespace) -> int:
    run_command = _strip_separator(args.command)
    if not run_command:
        run_command = ["pixi", "run", "python", "scripts/experiments/launch_matrix.py",
                       "--dry-run"]
    cmd = _base_run_cmd(args, interactive=args.tty)
    cmd += [args.image] + run_command
    return _run(cmd, dry_run=args.dry_run)


def _add_container_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--repo", type=Path, default=REPO)
    p.add_argument("--pixi-volume", default="opencda-marl-pixi-linux",
                   help="Named Docker volume mounted at .pixi inside the repo.")
    p.add_argument("--gpu", default="all",
                   help="GPU id, comma-separated ids, or 'all'.")
    p.add_argument("--network", default="host")
    p.add_argument("--user", default="1000:1000",
                   help="Container user for mounted-file writes; empty disables.")
    p.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command_name", required=True)

    build = sub.add_parser("build")
    build.add_argument("--image", default=DEFAULT_IMAGE)
    build.add_argument("--dockerfile", type=Path, default=DEFAULT_DOCKERFILE)
    build.add_argument("--context", type=Path, default=REPO)
    build.add_argument("--dry-run", action="store_true")
    build.set_defaults(func=cmd_build)

    shell = sub.add_parser("shell")
    _add_container_common(shell)
    shell.add_argument("--no-tty", action="store_true",
                       help="Disable -it for non-interactive CI sessions.")
    shell.set_defaults(func=cmd_shell)

    run = sub.add_parser("run")
    _add_container_common(run)
    run.add_argument("--tty", action="store_true",
                     help="Allocate a TTY for interactive commands.")
    run.add_argument("command", nargs=argparse.REMAINDER,
                     help="Command to execute after --.")
    run.set_defaults(func=cmd_run)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(_normalize_pixi_separator(argv or []))
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
