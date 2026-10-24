"""Carry actual collision identity through flag consumption, without retuning it."""
from collections import deque
from types import SimpleNamespace
import weakref

from opencda.core.safety.sensors import CollisionSensor
from opencda_marl._kernel.adapter.vehicle_adapter import MARLVehicleAdapter
from opencda_marl._kernel.sensing.marl_collision_sensor import MARLCollisionSensor


def test_collision_metadata_uses_the_sensor_event_not_the_nearest_vehicle():
    sensor = object.__new__(MARLCollisionSensor)
    sensor._history = deque(maxlen=5)
    sensor._threshold = 4.0
    sensor.collided = False
    event = SimpleNamespace(frame=99, normal_impulse=SimpleNamespace(x=3.0, y=4.0, z=0.0),
                            other_actor=SimpleNamespace(id=17, type_id='static.wall'))
    CollisionSensor._on_collision(weakref.ref(sensor), event)
    assert sensor.collided and sensor.collided_frame == 99

    adapter = object.__new__(MARLVehicleAdapter)
    adapter.actor_id = 7
    adapter.vm = SimpleNamespace(safety_manager=SimpleNamespace(sensors=[sensor]))
    assert adapter.check_collision() is True
    assert adapter.collision_details == {'frame': 99, 'other_actor_id': 17,
                                         'other_actor_type': 'static.wall', 'impulse': 5.0}
    assert adapter.collision_details is not sensor.last_collision_details
    assert adapter.check_collision() is False
    event.frame = 100
    event.normal_impulse = SimpleNamespace(x=4.0, y=0.0, z=0.0)
    CollisionSensor._on_collision(weakref.ref(sensor), event)
    assert sensor.collided is False, 'the original strict impulse threshold is unchanged'
    assert sensor.last_collision_details['frame'] == 99
