"""Resolve ONE consistent SUMO installation (binary + Python tools).

The failure this prevents: the environment ships ``eclipse-sumo`` (client
libs *and* binaries under ``site-packages/sumo``), but a system SUMO of
another version is also on the machine, named by ``SUMO_HOME`` or by a
``PYTHONPATH`` entry for its ``tools`` directory. Code that appended
``$SUMO_HOME/tools`` to ``sys.path`` and launched ``$SUMO_HOME/bin/sumo``
paired one version's ``traci`` client with the other's server, and every
``findRoute`` failed with "Retrieval of a route requires five parameter":
no SUMO vehicle ever spawned, and a training run silently learned from
nothing. Which client won was import-order dependent.

Policy: **prefer the environment's own SUMO** (guaranteed to match its
Python bindings); fall back to ``SUMO_HOME`` only when the package is
absent. The package's own ``tools`` directory (where ``eclipse-sumo``
ships ``traci`` and ``sumolib``; a plain pip install puts nothing else on
``sys.path``) goes first on ``sys.path``, and every other SUMO ``tools``
directory is dropped from it, whether ``SUMO_HOME`` or ``PYTHONPATH`` put
it there. Log the chosen versions once, and warn loudly if the client and
binary versions still disagree.

Import this module before anything imports ``traci``/``sumolib``.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

_RESOLVED: Optional[dict] = None


def _package_sumo_home() -> Optional[Path]:
    """The ``sumo`` PyPI package directory, if installed (has bin/ and tools/)."""
    try:
        import importlib.util

        spec = importlib.util.find_spec("sumo")
    except Exception:
        return None
    if spec is None or not spec.origin:
        return None
    root = Path(spec.origin).resolve().parent
    if (root / "bin").is_dir() and (root / "tools").is_dir():
        return root
    return None


def _path_key(entry: str) -> str:
    """A ``sys.path`` entry normalised for comparison (case, separators,
    trailing slash)."""
    return os.path.normcase(os.path.normpath(os.path.abspath(entry)))


def _is_sumo_tools_dir(entry) -> bool:
    """Whether a ``sys.path`` entry is a SUMO ``tools`` directory: a
    directory named ``tools`` that ships both ``traci/`` and ``sumolib/``.

    A ``site-packages`` holding the separate ``traci``/``sumolib``
    distributions, or a project's own ``tools/`` (even under a ``sumo*``
    folder), is never matched.
    """
    if not isinstance(entry, str) or not entry:
        return False
    path = Path(entry)
    if path.name.lower() != "tools":
        return False
    return (path / "traci").is_dir() and (path / "sumolib").is_dir()


def _drop_foreign_sumo_tools(keep: Path, env_home: Optional[str]) -> list:
    """Remove from ``sys.path`` every SUMO ``tools`` directory except ``keep``:
    the one ``SUMO_HOME`` names and any other (a ``PYTHONPATH`` entry for a
    system SUMO, say), so no other version's ``traci``/``sumolib`` can be
    imported. Returns the removed entries."""
    keep_key = _path_key(str(keep))
    env_tools_key = _path_key(str(Path(env_home) / "tools")) if env_home else None
    removed = []
    for entry in list(sys.path):
        if not isinstance(entry, str) or not entry:
            continue
        key = _path_key(entry)
        if key == keep_key:
            continue
        if key == env_tools_key or _is_sumo_tools_dir(entry):
            sys.path.remove(entry)
            removed.append(entry)
    return removed


def _binary_version(binary: Path) -> Optional[str]:
    try:
        out = subprocess.run([str(binary), "--version"], capture_output=True,
                             text=True, timeout=20).stdout
    except Exception:
        return None
    for token in out.replace("\n", " ").split():
        if token[0].isdigit() and token.count(".") >= 1:
            return token
    return None


def resolve_sumo_home() -> dict:
    """Pick the SUMO install, fix ``SUMO_HOME``/``sys.path`` accordingly, and
    return ``{"home", "source", "binary_version", "client_version"}``.

    Idempotent; the decision is made once per process.
    """
    global _RESOLVED
    if _RESOLVED is not None:
        return _RESOLVED

    env_home = os.environ.get("SUMO_HOME")
    pkg_home = _package_sumo_home()

    if pkg_home is not None:
        home, source = pkg_home, "eclipse-sumo package"
        if env_home and Path(env_home).resolve() != pkg_home:
            logger.warning(
                f"SUMO_HOME={env_home!r} points at a different SUMO than the "
                f"environment's eclipse-sumo package ({pkg_home}); using the "
                "package so client and server versions match. Unset SUMO_HOME "
                "to silence this.")
        # The package ships traci and sumolib under sumo/tools, which a plain
        # pip install does not put on sys.path: put it first, so they import
        # at all, and drop every other SUMO's tools directory (a stale
        # $SUMO_HOME/tools, or a PYTHONPATH entry for a system SUMO) so no
        # other version's bindings can be imported instead.
        pkg_tools = str(pkg_home / "tools")
        removed = _drop_foreign_sumo_tools(pkg_home / "tools", env_home)
        if removed:
            logger.info(
                "Dropped other SUMO tools directories from sys.path so the "
                f"bindings match the package: {', '.join(removed)}")
        # Exactly one copy of the package's tools, and it comes first.
        pkg_key = _path_key(pkg_tools)
        sys.path[:] = [entry for entry in sys.path
                       if not (isinstance(entry, str) and entry
                               and _path_key(entry) == pkg_key)]
        sys.path.insert(0, pkg_tools)
        os.environ["SUMO_HOME"] = str(pkg_home)
        os.environ.setdefault("PROJ_LIB", str(pkg_home / "data" / "proj"))
        os.environ.setdefault("PROJ_DATA", str(pkg_home / "data" / "proj"))
    elif env_home:
        home, source = Path(env_home), "SUMO_HOME"
        tools = str(home / "tools")
        if tools not in sys.path:
            sys.path.append(tools)
    else:
        raise RuntimeError(
            "No SUMO installation found: install eclipse-sumo into the "
            "environment (pixi install) or set SUMO_HOME.")

    binary = home / "bin" / ("sumo.exe" if os.name == "nt" else "sumo")
    binary_version = _binary_version(binary) if binary.exists() else None
    client_version = None
    try:
        import traci  # noqa: F401  (resolves after the sys.path fix above)

        client_version = getattr(traci, "__version__", None)
        if not client_version or client_version == "0.0.0":
            # traci read from the package's tools has no distribution metadata
            # of its own and reports the placeholder 0.0.0; it ships with the
            # binary, so the binary's version is the client's.
            client_version = _binary_version(binary) if binary.exists() else None
    except Exception:
        pass

    if binary_version and client_version and \
            binary_version.split(".")[:2] != client_version.split(".")[:2]:
        logger.warning(
            f"SUMO client/server version skew: traci {client_version} vs "
            f"binary {binary_version}; TraCI commands may be rejected.")

    _RESOLVED = {
        "home": str(home),
        "source": source,
        "binary": str(binary),
        "binary_version": binary_version,
        "client_version": client_version,
    }
    logger.info(f"SUMO resolved from {source}: {home} "
                f"(binary {binary_version or '?'})")
    return _RESOLVED


def sumo_binary(gui: bool = False) -> str:
    """Path to the resolved install's ``sumo``/``sumo-gui`` executable."""
    info = resolve_sumo_home()
    name = "sumo-gui" if gui else "sumo"
    if os.name == "nt":
        name += ".exe"
    return str(Path(info["home"]) / "bin" / name)


__all__ = ["resolve_sumo_home", "sumo_binary"]
