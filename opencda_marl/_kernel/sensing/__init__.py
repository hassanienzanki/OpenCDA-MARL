"""CARLA-side sensing the kernel owns: monitor, spectator, collision.

The future perception axis gets an ``opencda_marl.api`` contract; until
then these are kernel machinery.
"""

__all__ = ["CarlaMonitor", "CarlaSpectator", "MARLCollisionSensor"]


def __getattr__(name):
    targets = {
        "CarlaMonitor": "opencda_marl._kernel.sensing.carla_monitor",
        "CarlaSpectator": "opencda_marl._kernel.sensing.carla_spectator",
        "MARLCollisionSensor": "opencda_marl._kernel.sensing.marl_collision_sensor",
    }
    module = targets.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module), name)
