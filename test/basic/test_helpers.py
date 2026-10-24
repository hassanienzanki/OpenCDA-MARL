"""Shared test infrastructure for test/basic/.

This module is *not* a test file — it contains no ``test_`` functions.
Import it from sibling test modules to avoid duplicating CARLA geometry
stubs and the module-loader boilerplate.
"""

from pathlib import Path
import importlib.util
import sys

# ---------------------------------------------------------------------------
# Path bootstrapping (runs once at import time)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PIXI_SITE_PACKAGES = REPO_ROOT / ".pixi" / "envs" / "default" / "Lib" / "site-packages"
if PIXI_SITE_PACKAGES.is_dir() and str(PIXI_SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(PIXI_SITE_PACKAGES))


# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------

def load_module(name: str, relative_path: str):
    """Load a Python module by file path, relative to *REPO_ROOT*."""
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# CARLA geometry stubs (superset of all per-file variants)
# ---------------------------------------------------------------------------

class StubLocation:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def distance(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return (dx * dx + dy * dy + dz * dz) ** 0.5

    def __add__(self, other):
        return StubLocation(self.x + other.x, self.y + other.y, self.z + other.z)


class StubRotation:
    def __init__(self, yaw=0.0):
        self.yaw = yaw


class StubTransform:
    """Canonical constructor: ``StubTransform(location, yaw=0.0)``.

    Use :func:`make_transform` when you have raw ``(x, y, z, yaw)`` coords.
    """

    def __init__(self, location, yaw=0.0):
        self.location = location
        self.rotation = StubRotation(yaw)


def make_transform(x, y, z=0.0, yaw=0.0):
    """Convenience factory: ``make_transform(x, y, z, yaw)``."""
    return StubTransform(StubLocation(x, y, z), yaw)


class StubWaypoint:
    def __init__(self, x, y, yaw=0.0):
        self.transform = make_transform(x, y, 0.0, yaw)


class StubVelocity:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class StubVehicle:
    def __init__(self, location, velocity=None):
        self._location = location
        self._velocity = velocity or StubVelocity()

    def get_location(self):
        return self._location

    def get_velocity(self):
        return self._velocity


class StubWorld:
    """Minimal world stub with a debug recorder."""

    def __init__(self):
        self.debug = _DebugRecorder()

    def get_map(self):
        return object()


class _DebugRecorder:
    def __init__(self):
        self.points = []
        self.lines = []
        self.arrows = []

    def draw_point(self, location, size, color, life_time):
        self.points.append((location, size, color, life_time))

    def draw_line(self, begin, end, thickness, color, life_time):
        self.lines.append((begin, end, thickness, color, life_time))

    def draw_arrow(self, begin, end, thickness, arrow_size, color, life_time):
        self.arrows.append((begin, end, thickness, arrow_size, color, life_time))


# ---------------------------------------------------------------------------
# Config merge helper
# ---------------------------------------------------------------------------

def merged_scenario_config(scenario_ref):
    """Resolve + three-way merge a scenario config, return plain dict."""
    from omegaconf import OmegaConf

    config_merger = load_module("config_merger", "opencda_marl/utils/config_merger.py")
    config_locator = load_module("config_locator", "opencda_marl/utils/config_locator.py")

    resolved = config_locator.resolve_marl_config_paths(
        REPO_ROOT,
        scenario_ref,
        custom_config_roots=[REPO_ROOT / "opencda_marl" / "_bundled" / "configs"],
    )
    merged = config_merger.smart_merge(
        OmegaConf.load(resolved["default_yaml"]),
        OmegaConf.load(resolved["scenario_base_yaml"]),
    )
    merged = config_merger.smart_merge(merged, OmegaConf.load(resolved["scenario_yaml"]))
    return OmegaConf.to_container(merged, resolve=True)
