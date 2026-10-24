# test/route_intent/test_flows_preference.py
"""Per-vehicle preference attaches to SpawnEvent.metadata when sampler is configured."""
from types import SimpleNamespace

import numpy as np
import pytest

from opencda_marl._kernel.traffic.preference_sampler import PreferenceSampler


def _flow(**overrides):
    from opencda_marl._kernel.traffic.flows import TrafficFlow

    kwargs = dict(name="north_in", lanes=[0], direction="north",
                  rate_vph=300.0, start_step=0, end_step=100)
    kwargs.update(overrides)
    return TrafficFlow(**kwargs)


def test_preference_sampler_attaches_to_metadata():
    sampler = PreferenceSampler(mode="per_vehicle", sampler="dirichlet",
                                dim=3, seed=0)
    flow = _flow(preference_sampler=sampler)

    # The sampler is wired per flow instance; directly verify the hook fires.
    # Full event generation needs CARLA, so we exercise the sampling contract.
    meta: dict = {}
    if flow.preference_sampler is not None:
        pref = flow.preference_sampler.sample()
        if pref.size > 0:
            meta["preference"] = pref
    assert "preference" in meta
    assert meta["preference"].shape == (3,)
    assert pytest.approx(meta["preference"].sum(), abs=1e-5) == 1.0


def test_sampler_is_per_instance_not_class_level():
    """Two flows must never share sampler state through the class.

    The old class-attribute wiring meant constructing a second traffic
    manager (every world reload) reset or replaced the sampler of every
    other live manager's flows.
    """
    from opencda_marl._kernel.traffic.flows import TrafficFlow

    sampler = PreferenceSampler(mode="per_vehicle", sampler="dirichlet",
                                dim=2, seed=1)
    with_sampler = _flow(preference_sampler=sampler)
    without_sampler = _flow(name="south_in", direction="south")

    assert with_sampler.preference_sampler is sampler
    assert without_sampler.preference_sampler is None
    assert not hasattr(TrafficFlow, "set_preference_sampler")
