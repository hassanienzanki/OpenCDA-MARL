"""Config resolution and merging helpers.

Pure library utilities. The runnable maintenance tools (config lint, map
fixing, vehicle-type discovery) live in ``opencda_marl.tools`` and are run
as modules, not imported from here.
"""
from .config_merger import smart_merge
from .config_locator import resolve_marl_config_paths, MARL_CONFIG_FAMILIES

__all__ = [
    "smart_merge",
    "resolve_marl_config_paths",
    "MARL_CONFIG_FAMILIES",
]
