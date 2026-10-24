"""Environment facades — the public window onto the kernel's backends.

- ``MARLEnv`` / ``SumoMARLEnv``: the kernel's backend environments (their
  classes are reachable here for construction and tests; the episode loop
  itself is kernel-sealed)
- ``OpenCDAMARLParallelEnv``: the PettingZoo port — bring your own loop
- the dependency-free schema env (``traffic_schema_intersection``)
"""

__all__ = [
    "MARLEnv",
    "SumoMARLEnv",
    "OpenCDAMARLParallelEnv",
]


def __getattr__(name):
    if name == "MARLEnv":
        from opencda_marl._kernel.envs.marl_env import MARLEnv
        return MARLEnv
    if name == "SumoMARLEnv":
        from opencda_marl._kernel.envs.sumo_marl_env import SumoMARLEnv
        return SumoMARLEnv
    if name == "OpenCDAMARLParallelEnv":
        from opencda_marl.envs.parallel_env import OpenCDAMARLParallelEnv
        return OpenCDAMARLParallelEnv
    if name in ("marl_env", "sumo_marl_env", "sumo_runtime", "episode_state"):
        # Module forwarding: the backend modules live in the kernel; tests
        # and tooling reach them through this facade.
        import importlib

        return importlib.import_module(f"opencda_marl._kernel.envs.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
