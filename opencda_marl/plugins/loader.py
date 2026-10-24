"""Plugin discovery and loading — one mechanism for every kind.

Sources, in order: ``importlib.metadata`` entry points in the
``opencda_marl.plugins`` group (installed packages), the
``OPENCDA_MARL_PLUGINS`` environment variable (module names or ``.py``
paths, ``os.pathsep``-separated), the ``MARL.plugins`` list of a scenario
config, and the ``builtin/…/contrib`` drop-in folder (imported by the
built-in package itself). A plugin registers at import time — algorithms,
components and profiles alike — and may declare the API version it targets
as ``OPENCDA_MARL_POLICY_API``; on mismatch everything it registered, of
every kind, is rolled back and the plugin is reported as failed.
"""
from __future__ import annotations

import os
from typing import Dict

from opencda_marl.plugins import registry as _registry

#: ``importlib.metadata`` entry-point group external packages advertise.
PLUGIN_ENTRY_POINT_GROUP = "opencda_marl.plugins"
#: ``os.pathsep``-separated module names or ``.py`` paths to import.
PLUGIN_ENV = "OPENCDA_MARL_PLUGINS"

#: The engine's plugin-contract version. Bump when ``Policy`` /
#: ``PolicyCapabilities``, a kind's constructor signature, or the factory
#: signatures change incompatibly.
POLICY_API_VERSION = 1

_plugins_loaded = False
_plugin_failures: Dict[str, str] = {}

#: Keys of the plugins already imported (absolute path for a ``.py``
#: file, the module name otherwise). A run loads ``MARL.plugins`` from the
#: coordinator and again from the manager; re-executing a file plugin would
#: register its components twice and report a working plugin as failed.
_loaded_extras: set = set()


def _spec_key(spec: str) -> str:
    """Identity of a plugin spec: a file by its absolute path, a module by name."""
    if spec.endswith(".py") or os.sep in spec or "/" in spec:
        return os.path.normcase(os.path.abspath(os.path.expanduser(spec)))
    return spec


def plugin_failures() -> Dict[str, str]:
    return dict(_plugin_failures)


def _import_plugin(spec: str) -> None:
    """Import one plugin; registrations of every kind run at import time."""
    import importlib
    import importlib.util
    import sys

    _registry.current_source = spec
    try:
        if spec.endswith(".py") or os.sep in spec or "/" in spec:
            path = os.path.abspath(os.path.expanduser(spec))
            name = "opencda_marl_plugin_" + os.path.splitext(os.path.basename(path))[0]
            module_spec = importlib.util.spec_from_file_location(name, path)
            if module_spec is None or module_spec.loader is None:
                raise ImportError(f"cannot load plugin file {path}")
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[name] = module
            module_spec.loader.exec_module(module)
        else:
            module = importlib.import_module(spec)
        declared = getattr(module, "OPENCDA_MARL_POLICY_API", None)
        if declared is not None and int(declared) != POLICY_API_VERSION:
            removed = _registry.rollback(spec)
            raise ImportError(
                f"targets plugin API v{int(declared)}, this engine provides "
                f"v{POLICY_API_VERSION} — {removed} registration(s) rolled back")
    finally:
        _registry.current_source = None


def load_plugins(extra: "list[str] | tuple[str, ...] | str | None" = None, *,
                 force: bool = False) -> Dict[str, str]:
    """Import every external plugin package; returns ``{plugin: error}``.

    Entry points and the environment are read once per process unless
    ``force``; ``extra`` (a config's ``MARL.plugins`` list) is applied on
    every call, but a plugin that already imported successfully (a file by
    absolute path, a module by name) is not imported again unless ``force``. A failure is logged and re-surfaced by kind resolvers when an
    unknown key is requested, so a broken plugin cannot hide behind
    "unknown component".
    """
    from loguru import logger

    global _plugins_loaded
    sources: list[str] = []
    if force or not _plugins_loaded:
        _plugins_loaded = True
        try:
            from importlib.metadata import entry_points
            eps = entry_points()
            group = eps.select(group=PLUGIN_ENTRY_POINT_GROUP) if hasattr(eps, "select") \
                else eps.get(PLUGIN_ENTRY_POINT_GROUP, [])
            for ep in group:
                sources.append(ep.value.split(":")[0])
        except Exception as exc:  # noqa: BLE001 - metadata problems must not stop training
            logger.debug(f"entry-point scan failed: {exc}")
        env = os.environ.get(PLUGIN_ENV, "")
        sources.extend(v.strip() for v in env.split(os.pathsep) if v.strip())
    if isinstance(extra, str):
        extra = [extra]
    sources.extend(str(v) for v in (extra or []))
    for spec in sources:
        key = _spec_key(spec)
        if key in _loaded_extras and not force:
            continue
        try:
            _import_plugin(spec)
            _plugin_failures.pop(spec, None)
            _loaded_extras.add(key)
            logger.info(f"plugin loaded: {spec}")
        except Exception as exc:  # noqa: BLE001
            _plugin_failures[spec] = f"{type(exc).__name__}: {exc}"
            logger.warning(f"plugin {spec!r} failed to import: {exc}")
    return dict(_plugin_failures)
