"""Shared utilities for the direction-preserving matrix launchers."""
from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from omegaconf import OmegaConf


# A cell slug is ``<prefix><map>_<condition>_<regime>_seed<N>``; the prefix
# is the launcher's ``--slug-prefix``.
DEFAULT_SLUG_PREFIX = "matrix_"
DEFAULT_MAP = "custom"
# The prefix as the analyzers match it: any prefix, not only the current
# default, so cells launched with another ``--slug-prefix`` (or under an
# earlier default) still parse instead of silently dropping out of a table.
SLUG_PREFIX_PATTERN = r"(?P<prefix>.*?)"


def mixed_prefix_warning(prefixes: Iterable[str]) -> str | None:
    """The warning for cells of several runs loaded together, or None.

    One root can hold the cells of several runs (one per ``--slug-prefix``);
    the analyzers keep the prefix in every row and group, and pass
    ``--slug-prefix`` to read a single run.
    """
    distinct = sorted(set(prefixes))
    if len(distinct) < 2:
        return None
    shown = ", ".join(repr(p) for p in distinct)
    return (f"WARN: cells from {len(distinct)} slug prefixes ({shown}); "
            "pass --slug-prefix to analyze one run")


def preference_block_path(cfg: Any) -> str | None:
    """Config path of the spawn-time preference block, or None.

    The canonical home is ``agents.preference``; the historical
    ``agents.route_intent.preference`` still resolves. Resolving it here
    rather than assuming one location keeps the launchers' regime and seed
    patching working on both — a launcher that patches the wrong path
    silently runs every cell in the profile's default regime.
    """
    agents = cfg.get("agents") if hasattr(cfg, "get") else None
    if agents is None:
        return None
    if hasattr(agents, "get") and agents.get("preference") is not None:
        return "agents.preference"
    route_intent = agents.get("route_intent") if hasattr(agents, "get") else None
    if route_intent is not None and hasattr(route_intent, "get") \
            and route_intent.get("preference") is not None:
        return "agents.route_intent.preference"
    return None
MARL_CONFIG_ROOTS_ENV = "OPENCDA_MARL_CONFIG_ROOTS"
CANONICAL_ALGORITHMS = {
    "matd3",
    "masac",
    "mappo",
    "pcmatd3",
    "pcmasac",
    "pcmappo",
}


@dataclass(frozen=True)
class MapProfile:
    slug: str
    town: str
    xodr_path: str | None
    active_junctions: tuple[int, ...] | None
    replay_file: str
    spectator: dict[str, Any] | None = None


MAP_PROFILES: dict[str, MapProfile] = {
    "custom": MapProfile(
        slug="custom",
        town="intersection_local_xodr",
        xodr_path="opencda_marl/assets/intersection/intersection.xodr",
        active_junctions=(4,),
        replay_file="recordings/traffic_200vph.json",
    ),
}


def canonical_map(map_slug: str) -> str:
    """The ``MAP_PROFILES`` key a map argument names, matched case-insensitively.

    The launchers name their cells with this key, never the raw argument:
    run_matrix_eval and the analyzers match the map in a slug exactly, so
    ``--maps Custom`` writing ``matrix_Custom_*`` cells left them out of
    every evaluation and table. Raises ``ValueError`` for an unknown map.
    """
    key = str(map_slug).lower()
    if key not in MAP_PROFILES:
        raise ValueError(
            f"unknown map profile {map_slug!r}; valid: {sorted(MAP_PROFILES)}"
        )
    return key


def canonical_maps(map_slugs: Iterable[str]) -> list[str]:
    """``canonical_map`` of each map argument, in order, each key once.

    ``--maps custom Custom`` names one map twice; planning both would run
    the same cell slug twice, or in two parallel slots at once.
    """
    return list(dict.fromkeys(canonical_map(m) for m in map_slugs))


def map_profile(map_slug: str) -> MapProfile:
    return MAP_PROFILES[canonical_map(map_slug)]


def validate_matrix_schema(cfg: Any, *, source: str) -> None:
    """Reject the legacy config schema before runtime YAML is generated."""
    marl = cfg.get("MARL") if hasattr(cfg, "get") else None
    if marl is None:
        return
    if marl.get("intent_policy") is not None:
        raise ValueError(
            f"{source}: MARL.intent_policy is not allowed in active configs; "
            "use MARL.action_space."
        )
    algorithm = str(marl.get("algorithm", ""))
    if algorithm not in CANONICAL_ALGORITHMS:
        raise ValueError(
            f"{source}: non-canonical MARL.algorithm={algorithm!r}; expected one of "
            f"{sorted(CANONICAL_ALGORITHMS)}."
        )
    if marl.get("action_space") is None:
        raise ValueError(f"{source}: MARL.action_space is required.")


def slug_for(
    *,
    map_slug: str,
    condition: str,
    regime: str,
    seed: int,
    prefix: str = DEFAULT_SLUG_PREFIX,
    legacy: bool = False,
) -> str:
    """Return the artifact slug for one matrix cell.

    Prefixed slugs always include the map dimension. ``legacy=True``
    returns the former unprefixed form, kept for backward compatibility.
    """
    if legacy:
        return f"{condition}_{regime}_seed{int(seed)}"
    return f"{prefix}{map_slug}_{condition}_{regime}_seed{int(seed)}"


def patch_map_profile(
    cfg: Any,
    map_slug: str,
    *,
    active_junctions: Iterable[int] | None = None,
    replay_file: str | None = None,
    carla_rpc_port: int | None = None,
) -> Any:
    """Patch an OmegaConf config with map and CARLA port settings."""
    profile = map_profile(map_slug)
    OmegaConf.update(cfg, "scenario.map.town", profile.town,
                     merge=True, force_add=True)
    OmegaConf.update(cfg, "scenario.map.xodr_path", profile.xodr_path,
                     merge=True, force_add=True)
    junctions = (
        tuple(int(v) for v in active_junctions)
        if active_junctions is not None
        else profile.active_junctions
    )
    if junctions is not None:
        OmegaConf.update(cfg, "scenario.traffic.active_junctions", list(junctions),
                         merge=True, force_add=True)
    if replay_file is not None:
        OmegaConf.update(cfg, "scenario.traffic.replay_file", replay_file,
                         merge=True, force_add=True)
    elif profile.replay_file:
        OmegaConf.update(cfg, "scenario.traffic.replay_file", profile.replay_file,
                         merge=True, force_add=True)
    if profile.spectator is not None:
        for key, value in profile.spectator.items():
            OmegaConf.update(cfg, f"scenario.spectator.{key}", value,
                             merge=True, force_add=True)
    if carla_rpc_port is not None:
        OmegaConf.update(cfg, "world.client_port", int(carla_rpc_port),
                         merge=True, force_add=True)
    return cfg


def resolve_carla_binary(
    *,
    carla_root: str | None = None,
    carla_bin: str | None = None,
) -> Path:
    """Resolve the CARLA executable from CLI args and the environment.

    An explicit binary (``carla_bin`` or ``CARLA_BIN``) wins; otherwise the
    platform's launcher under ``carla_root`` or ``CARLA_ROOT``. There is no
    install-path guess: with neither set this raises ``ValueError``, because
    a guessed path only fails later, when a crashed server cannot restart.
    """
    explicit_bin = carla_bin or os.environ.get("CARLA_BIN")
    if explicit_bin:
        return Path(explicit_bin)

    root_value = carla_root or os.environ.get("CARLA_ROOT")
    if not root_value:
        raise ValueError(
            "CARLA binary unknown: set CARLA_ROOT (or CARLA_BIN), or pass "
            "--carla-root / --carla-bin; use --carla-restart-mode none when "
            "CARLA is managed outside the launcher."
        )
    root = Path(root_value)
    if platform.system().lower().startswith("win"):
        return root / "CarlaUE4.exe"
    return root / "CarlaUE4.sh"


def carla_process_names() -> tuple[str, ...]:
    if platform.system().lower().startswith("win"):
        return ("CarlaUE4.exe", "CarlaUE4-Win64-Shipping.exe")
    return ("CarlaUE4", "CarlaUE4-Linux-Shipping")


def prepend_marl_config_root(env: dict[str, str], config_root: Path) -> None:
    """Add a private config root for subprocess MARL config resolution."""
    root_value = str(config_root)
    existing = env.get(MARL_CONFIG_ROOTS_ENV)
    if not existing:
        env[MARL_CONFIG_ROOTS_ENV] = root_value
        return
    parts = [part for part in existing.split(os.pathsep) if part]
    if root_value not in parts:
        env[MARL_CONFIG_ROOTS_ENV] = os.pathsep.join([root_value, *parts])
