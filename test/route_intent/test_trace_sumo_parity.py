"""The step trace runs on the SUMO backend: no scenario manager, so it reads
the environment's own post-step observations and events."""
import json
from types import SimpleNamespace

from opencda_marl.api.events import StepEvent
from opencda_marl.telemetry.trace import RouteActionTraceWriter


def test_record_step_reads_the_sumo_env_state(tmp_path):
    env = SimpleNamespace(
        # the raw post-step pass keeps population and learning; the
        # normalised copy the env also holds would have dropped them
        trace_observations={"v1": {
            "location": {"x": 1.5, "y": -2.0}, "speed": 12.0, "heading_angle": 0.3,
            "velocity": {"x": 3.0, "y": 1.0}, "population": "background", "learning": False}},
        previous_observations={"v1": {"location": {"x": 1.5, "y": -2.0}, "speed": 12.0}},
        events=[StepEvent(step=3, event_id="e1", vehicle_id="v1", event_type="success")])
    coordinator = SimpleNamespace(states={"episode": 2, "step": 4}, marl_env=env,
                                  config={"scenario": {"type": "intersection"}})
    writer = RouteActionTraceWriter(
        {"enabled": True, "dir": str(tmp_path / "t"), "include_routes": False}, coordinator)
    writer.record_step(coordinator)
    writer.close()
    lines = (tmp_path / "t" / "steps.jsonl").read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[-1])
    assert record["episode"] == 2 and record["step"] == 3
    agent = record["agents"][0]
    assert agent["agent_id"] == "v1" and agent["x"] == 1.5 and agent["y"] == -2.0
    assert agent["population"] == "background" and agent["learning"] is False
    assert record["events"][0]["event_type"] == "success"


def _tf():
    return SimpleNamespace(location=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                           rotation=SimpleNamespace(pitch=0.0, yaw=0.0, roll=0.0))


def test_the_sumo_trace_keys_on_the_env_counters_and_finds_its_planner(tmp_path):
    route = {"entry_direction": "north", "exit_direction": "south", "entry_lane_idx": 0,
             "exit_lane_idx": 0, "route_type": "straight", "spawn_tf": _tf(), "dest_tf": _tf()}
    planner = SimpleNamespace(_junctions={4: {"routes": {1: route}}})
    env = SimpleNamespace(previous_observations={"v1": {"location": {"x": 0.0, "y": 0.0}}},
                          events=[], current_episode=2, current_step=4,
                          traffic_manager=SimpleNamespace(planner=planner))
    coordinator = SimpleNamespace(states={"episode": 0, "step": 0}, marl_env=env, config={})
    writer = RouteActionTraceWriter({"enabled": True, "dir": str(tmp_path / "t")}, coordinator)
    writer.record_step(coordinator)
    env.current_step = 5
    writer.record_step(coordinator)
    writer.close()
    records = [json.loads(line) for line in
               (tmp_path / "t" / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(r["episode"], r["step"]) for r in records] == [(2, 3), (2, 4)]
    routes = json.loads((tmp_path / "t" / "routes.json").read_text(encoding="utf-8"))
    assert routes and routes[0]["junction_id"] == 4 and routes[0]["route_id"] == 1
