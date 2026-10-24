"""WorldSnapshot.capture skips an actor it cannot read instead of taking the run down.

An actor that dies between the listing and the read, or a simulator that
stops answering for one call, is expected; the capture is meant to skip
it with a warning and carry on. Its error path referred to a logger the
module never imported, so the skip itself raised NameError and ended the
run.
"""
from types import SimpleNamespace

from loguru import logger

from opencda_marl._kernel.adapter.world_snapshot import WorldSnapshot


class _Vector:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


class _Actor:
    def __init__(self, actor_id, x, y, broken=False):
        self.id = actor_id
        self.is_alive = True
        self.type_id = "vehicle.test"
        self._broken = broken
        self._transform = SimpleNamespace(location=_Vector(x, y), rotation=SimpleNamespace(yaw=0.0))

    def get_transform(self):
        if self._broken:
            raise RuntimeError("time-out of 20000ms while waiting for the simulator")
        return self._transform

    def get_velocity(self):
        return _Vector(1.0, 0.0)


class _World:
    def __init__(self, actors):
        self._actors = actors

    def get_actors(self):
        return SimpleNamespace(filter=lambda pattern: list(self._actors))


def test_an_unreadable_actor_is_skipped_with_a_warning_and_the_rest_captured():
    world = _World([_Actor(1, 0.0, 0.0), _Actor(2, 10.0, 0.0, broken=True), _Actor(3, 20.0, 0.0)])
    messages = []
    sink = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING")
    try:
        snapshot = WorldSnapshot.capture(world)
    finally:
        logger.remove(sink)
    assert sorted(v.actor_id for v in snapshot.vehicles) == [1, 3]
    assert any("skipped 1 actor" in m for m in messages)
