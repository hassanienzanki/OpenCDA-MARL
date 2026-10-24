"""Shared pytest setup for the route-intent test suite.

Ensures the repository root is importable regardless of how pytest is
invoked, then imports ``torch`` before ``numpy`` to avoid DLL load-order
errors on Windows pixi envs.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: F401,E402  (import-order side effect)
