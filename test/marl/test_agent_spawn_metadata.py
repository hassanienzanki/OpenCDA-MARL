"""Spawn metadata reaches the agent factory, so a driver model can be
per-vehicle and reproducible from the recording."""
def test_the_agent_factory_sees_spawn_metadata_only_when_present():
    from opencda_marl._kernel.adapter.vehicle_adapter import MARLVehicleAdapter

    adapter = object.__new__(MARLVehicleAdapter)
    adapter.config = {"agent_type": "marl", "marl": {"max_speed": 45}}
    adapter.spawn_metadata = {}
    assert adapter._agent_config() is adapter.config
    adapter.spawn_metadata = {"route_id": 7, "driver": {"tau": 1.2}}
    view = adapter._agent_config()
    assert view is not adapter.config
    assert view["marl"] == {"max_speed": 45}
    assert view["spawn_metadata"] == {"route_id": 7, "driver": {"tau": 1.2}}
    assert "spawn_metadata" not in adapter.config
