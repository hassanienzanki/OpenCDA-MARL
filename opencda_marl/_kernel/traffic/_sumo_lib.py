"""Single-source SUMO Python binding selector.

Provides a ``traci`` name that resolves to either :mod:`libsumo` (default, fast
shared-memory) or :mod:`traci` (fallback, TCP socket — needed when using
``sumo-gui``). All in-repo SUMO call sites should do::

    from opencda_marl._kernel.traffic._sumo_lib import traci

instead of ``import traci`` directly, so the binding switch is one knob.

Selection:

* ``OPENCDA_SUMO_LIB=libsumo`` (default if env unset) → :mod:`libsumo`. ~2-5×
  faster per call than ``traci`` because it skips the TCP socket roundtrip
  and calls the SUMO C++ library directly through CPython. Only works with
  headless ``sumo`` (no GUI), which is what training and headless eval use.
* ``OPENCDA_SUMO_LIB=traci`` → :mod:`traci`. Slower but supports ``sumo-gui``
  for interactive debugging.

Falls back to ``traci`` automatically if ``libsumo`` import fails (e.g.,
old SUMO install without the shared-memory bindings).
"""
from __future__ import annotations

import os
import warnings

# Establish one consistent SUMO install (binary + tools) BEFORE the bindings
# are imported, so client and server versions cannot diverge.
from opencda_marl._kernel.traffic._sumo_home import resolve_sumo_home  # noqa: E402

_SUMO_INFO = resolve_sumo_home()

_PREF = os.environ.get("OPENCDA_SUMO_LIB", "libsumo").strip().lower()

if _PREF not in ("libsumo", "traci"):
    warnings.warn(
        f"OPENCDA_SUMO_LIB={_PREF!r} not recognised; defaulting to 'libsumo'.",
        RuntimeWarning,
        stacklevel=2,
    )
    _PREF = "libsumo"

if _PREF == "libsumo":
    try:
        import libsumo as traci  # type: ignore  # noqa: F401
        SUMO_BINDING = "libsumo"
    except ImportError as exc:
        # libsumo is its own PyPI package, versioned with SUMO itself: name
        # the version that matches the resolved binary.
        _version = (_SUMO_INFO or {}).get("binary_version") or "<your SUMO version>"
        warnings.warn(
            f"libsumo import failed ({exc}); falling back to traci, which is "
            f"slower per simulation step. Install the matching binding with "
            f"`pip install libsumo=={_version}` (a pixi environment declares "
            "it: run `pixi install`), or set OPENCDA_SUMO_LIB=traci to silence.",
            RuntimeWarning,
            stacklevel=2,
        )
        import traci  # type: ignore  # noqa: F401
        SUMO_BINDING = "traci"
else:
    import traci  # type: ignore  # noqa: F401
    SUMO_BINDING = "traci"

__all__ = ["traci", "SUMO_BINDING"]
