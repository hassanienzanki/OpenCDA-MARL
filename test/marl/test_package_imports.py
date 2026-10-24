"""Public package imports must work in any order.

``opencda_marl._kernel.envs.marl_env`` imports ``lifecycle.scenario_manager``, and
``lifecycle.coordinator`` imports ``opencda_marl.envs`` back. While
``lifecycle/__init__`` imported the coordinator eagerly, the first of the two
packages a process touched decided whether the other one imported at all:
``from opencda_marl.envs import MARLEnv`` in a fresh interpreter raised
``ImportError: cannot import name 'MARLEnv' from partially initialized
module``. Only ``opencda.py``'s import order hid it.
"""
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# torch first: the Windows pixi env needs that DLL load order (see conftest).
ORDERS = {
    "envs_first": "from opencda_marl.envs import MARLEnv, SumoMARLEnv",
    "lifecycle_first": "from opencda_marl._kernel.lifecycle import MARLCoordinator, MARLScenarioManager",
    "env_module_first": "import opencda_marl._kernel.envs.marl_env as m; assert m.MARLEnv",
    "scenario_module_first": ("import opencda_marl._kernel.lifecycle.scenario_manager as s;"
                              " from opencda_marl.envs import MARLEnv"),
}


@pytest.mark.parametrize("order", sorted(ORDERS), ids=sorted(ORDERS))
def test_import_order_does_not_break_the_public_names(order):
    code = chr(10).join(["import torch", ORDERS[order], "print('ok')"])
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, cwd=str(REPO), timeout=300)
    assert result.returncode == 0 and "ok" in result.stdout, (
        "import order " + order + " failed:" + chr(10) + result.stderr[-1500:])


def test_lifecycle_exports_are_complete():
    import opencda_marl._kernel.lifecycle as lifecycle

    for name in lifecycle.__all__:
        assert getattr(lifecycle, name) is not None
    assert dir(lifecycle) == sorted(lifecycle.__all__)
    with pytest.raises(AttributeError):
        lifecycle.NoSuchThing
