"""Manage a CARLA 0.9.16 server container for OpenCDA-MARL.

This helper is for the default server-only Docker workflow: CARLA runs in one
container, while training runs on the host or in a separate bind-mounted
training runner. It is intentionally small and uses the official CARLA image by
default.
"""
from __future__ import annotations

import argparse
import platform
import shlex
import subprocess
import sys
from typing import Sequence


DEFAULT_IMAGE = "carlasim/carla:0.9.16"
HELPER_OPTIONS = {
    "--dry-run",
    "--gpu",
    "--image",
    "--name",
    "--network",
    "--no-privileged",
    "--no-follow",
    "--rpc-port",
    "--streaming-port",
    "--tail",
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


def _run(cmd: Sequence[str], *, dry_run: bool = False, check: bool = True) -> int:
    print(f"[docker-carla] $ {_quote(cmd)}", flush=True)
    if dry_run:
        return 0
    completed = subprocess.run(list(cmd), check=False)
    if check and completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return int(completed.returncode)


def _inspect(name: str, template: str) -> str | None:
    completed = subprocess.run(
        ["docker", "inspect", "-f", template, name],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _container_exists(name: str) -> bool:
    return _inspect(name, "{{.Id}}") is not None


def _container_running(name: str) -> bool:
    return _inspect(name, "{{.State.Running}}") == "true"


def _gpu_args(gpu: str) -> list[str]:
    gpu = str(gpu).strip()
    if not gpu or gpu.lower() == "all":
        return ["--gpus", "all", "-e", "NVIDIA_VISIBLE_DEVICES=all"]
    return ["--gpus", f"device={gpu}", "-e", f"NVIDIA_VISIBLE_DEVICES={gpu}"]


def _warn_host_network(network: str) -> None:
    if network == "host" and platform.system().lower().startswith("win"):
        print(
            "[docker-carla] warning: Docker Desktop on Windows does not provide "
            "Linux-style --network host. This workflow is intended for Ubuntu "
            "servers; on Windows, publish ports manually or run CARLA directly.",
            flush=True,
        )


def cmd_pull(args: argparse.Namespace) -> int:
    return _run(["docker", "pull", args.image], dry_run=args.dry_run)


def cmd_start(args: argparse.Namespace) -> int:
    name = args.name or f"opencda-carla-{args.rpc_port}"
    extra_args = _strip_separator(args.extra_args)
    _warn_host_network(args.network)
    if _container_running(name):
        print(f"[docker-carla] already running: {name}", flush=True)
        return 0
    if _container_exists(name):
        _run(["docker", "rm", "-f", name], dry_run=args.dry_run)

    carla_cmd = [
        "bash",
        "CarlaUE4.sh",
        "-RenderOffScreen",
        "-nosound",
        f"-carla-rpc-port={int(args.rpc_port)}",
        f"-carla-streaming-port={int(args.streaming_port)}",
    ] + extra_args
    # Use --restart unless-stopped instead of --rm. --rm auto-removes the
    # container on ANY exit (including CARLA crashes), which is catastrophic
    # for unattended multi-hour training: if CARLA dies, the container
    # disappears and `--carla-restart-mode none` in the launcher has nothing
    # to fall back to. --restart unless-stopped keeps the container around
    # and lets docker auto-restart CARLA on crash, so training survives
    # transient CARLA failures. To explicitly remove a container created
    # this way: `pixi run carla-docker-stop && docker rm opencda-carla-2000`.
    cmd = [
        "docker",
        "run",
        "--restart",
        "unless-stopped",
        "-d",
        "--name",
        name,
        "--network",
        args.network,
    ]
    if args.privileged:
        cmd.append("--privileged")
    cmd += _gpu_args(args.gpu)
    cmd += [
        "-e",
        "NVIDIA_DRIVER_CAPABILITIES=all",
        args.image,
    ] + carla_cmd
    return _run(cmd, dry_run=args.dry_run)


def cmd_stop(args: argparse.Namespace) -> int:
    name = args.name or f"opencda-carla-{args.rpc_port}"
    if not _container_exists(name):
        print(f"[docker-carla] not found: {name}", flush=True)
        return 0
    return _run(["docker", "stop", name], dry_run=args.dry_run)


def cmd_status(args: argparse.Namespace) -> int:
    name = args.name or f"opencda-carla-{args.rpc_port}"
    cmd = [
        "docker",
        "ps",
        "-a",
        "--filter",
        f"name=^{name}$",
        "--format",
        "table {{.Names}}\t{{.Status}}\t{{.Image}}\t{{.Ports}}",
    ]
    return _run(cmd, dry_run=args.dry_run, check=False)


def cmd_logs(args: argparse.Namespace) -> int:
    name = args.name or f"opencda-carla-{args.rpc_port}"
    cmd = ["docker", "logs", "--tail", str(args.tail)]
    if args.follow:
        cmd.append("-f")
    cmd.append(name)
    return _run(cmd, dry_run=args.dry_run, check=False)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--name", default=None,
                   help="Container name. Defaults to opencda-carla-<rpc-port>.")
    p.add_argument("--rpc-port", type=int, default=2000)
    p.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    pull = sub.add_parser("pull")
    pull.add_argument("--image", default=DEFAULT_IMAGE)
    pull.add_argument("--dry-run", action="store_true")
    pull.set_defaults(func=cmd_pull)

    start = sub.add_parser("start")
    _add_common(start)
    start.add_argument("--streaming-port", type=int, default=0)
    start.add_argument("--gpu", default="all",
                       help="GPU id, comma-separated ids, or 'all'.")
    start.add_argument("--network", default="host")
    start.add_argument("--no-privileged", dest="privileged",
                       action="store_false")
    start.set_defaults(privileged=True, func=cmd_start)
    start.add_argument("extra_args", nargs=argparse.REMAINDER,
                       help="Extra CARLA args after --.")

    stop = sub.add_parser("stop")
    _add_common(stop)
    stop.set_defaults(func=cmd_stop)

    status = sub.add_parser("status")
    _add_common(status)
    status.set_defaults(func=cmd_status)

    logs = sub.add_parser("logs")
    _add_common(logs)
    logs.add_argument("--tail", type=int, default=80)
    logs.add_argument("--no-follow", dest="follow", action="store_false")
    logs.set_defaults(follow=True, func=cmd_logs)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(_normalize_pixi_separator(argv or []))
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
