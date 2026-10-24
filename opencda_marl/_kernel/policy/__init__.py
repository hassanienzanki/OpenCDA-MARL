"""Policy-driving machinery: manager, factory, strategies, checkpoints.

``MARLManager`` resolves lazily to keep the torch-vs-numpy import order
free (see test/conftest.py).
"""

__all__ = ["MARLManager"]


def __getattr__(name):
    if name == "MARLManager":
        from opencda_marl._kernel.policy.manager import MARLManager
        return MARLManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
