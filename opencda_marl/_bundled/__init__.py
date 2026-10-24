"""Data bundled into the engine wheel — and the repository's own source for
it: every config family (``marl``, ``pcma``, ``momarl``, ``sumo-marl``,
``cosim-marl`` and the classic ``opencda`` scenarios), the deterministic
traffic recordings, and (via ``opencda_marl/assets``) the intersection SUMO
network. A pip install and a checkout resolve the same files.

The resolution order everywhere is: explicit roots (``--marl-config-root``,
``OPENCDA_MARL_CONFIG_ROOTS``) first, a checkout-level ``configs/`` if one
exists, this bundle last — so an external root only ever *adds* or
overrides, and the launchers' per-cell runtime profiles live under
``artifacts/_runtime`` as such a root. ``test/basic/test_bundled_data.py``
pins the layout.
"""
from pathlib import Path

BUNDLED_ROOT = Path(__file__).resolve().parent
BUNDLED_CONFIG_ROOT = BUNDLED_ROOT / "configs"
BUNDLED_RECORDINGS = BUNDLED_ROOT / "recordings"
BUNDLED_CHECKPOINTS = BUNDLED_ROOT / "checkpoints"


def bundled_checkpoint(name: str) -> Path | None:
    """A packaged checkpoint by kit-relative name, or None.

    ``ral/sac_300vph`` names a run directory (its ``checkpoint.pth``);
    ``ral/sac_300vph/checkpoint.pth`` names the file. Absolute paths and
    parent references never resolve here.
    """
    rel = Path(str(name).replace("\\", "/"))
    if rel.is_absolute() or ".." in rel.parts:
        return None
    candidate = BUNDLED_CHECKPOINTS / rel
    if candidate.is_dir():
        candidate = candidate / "checkpoint.pth"
    return candidate if candidate.is_file() else None


def bundled_config_root() -> Path | None:
    """The packaged config root, or None when the bundle is absent."""
    return BUNDLED_CONFIG_ROOT if BUNDLED_CONFIG_ROOT.is_dir() else None


def bundled_recording(name: str) -> Path | None:
    """A packaged recording by basename, or None."""
    candidate = BUNDLED_RECORDINGS / Path(name).name
    return candidate if candidate.is_file() else None
