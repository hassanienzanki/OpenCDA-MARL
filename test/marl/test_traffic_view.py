"""The schematic traffic view: publisher files + monitor routes.

The publisher must write atomically into the run's metrics directory and
throttle by wall clock; the monitor must serve the two files under the
run's API namespace and 404 cleanly when a run has no view.
"""
import json

import pytest

from opencda_marl.telemetry.monitor import MonitorApp
from opencda_marl.telemetry.store import MetricsStore
from opencda_marl.telemetry.traffic_view import (
    GEOMETRY_NAME,
    STATE_NAME,
    TrafficViewPublisher,
)


def _vehicles():
    return [
        {"id": "rl_1", "x": 10.0, "y": -3.0, "yaw": 1.57, "speed": 8.2, "role": "agent"},
        {"id": "bg_7", "x": -22.5, "y": 14.0, "yaw": 0.0, "speed": 0.0, "role": "traffic"},
    ]


def test_publisher_writes_both_files(tmp_path):
    pub = TrafficViewPublisher(tmp_path / "run", interval=0.05)
    assert pub.due()
    wrote = pub.publish_state(_vehicles(), episode=3, step=120)
    assert wrote
    pub.publish_geometry({"lanes": [[[0, 0], [50, 0]]], "junction": [25, 0], "y_up": True})

    state = json.loads((tmp_path / "run" / STATE_NAME).read_text())
    assert state["episode"] == 3 and state["step"] == 120
    assert {v["role"] for v in state["vehicles"]} == {"agent", "traffic"}
    assert state["wall"] > 0

    geometry = json.loads((tmp_path / "run" / GEOMETRY_NAME).read_text())
    assert geometry["lanes"] == [[[0, 0], [50, 0]]]
    assert geometry["y_up"] is True


def test_publisher_throttles_and_reports(tmp_path):
    pub = TrafficViewPublisher(tmp_path, interval=60.0)
    assert pub.publish_state([], episode=0, step=0)
    # Inside the interval: not due, publish is a no-op returning False.
    assert not pub.due()
    assert not pub.publish_state(_vehicles(), episode=0, step=1)
    state = json.loads((tmp_path / STATE_NAME).read_text())
    assert state["step"] == 0 and state["vehicles"] == []


def test_geometry_is_write_once_until_invalidated(tmp_path):
    pub = TrafficViewPublisher(tmp_path, interval=0.05)
    pub.publish_geometry({"lanes": [[[0, 0], [1, 1]]]})
    pub.publish_geometry({"lanes": [[[9, 9], [8, 8]]]})   # ignored
    assert json.loads((tmp_path / GEOMETRY_NAME).read_text())["lanes"][0][0] == [0, 0]
    pub.invalidate_geometry()
    pub.publish_geometry({"lanes": [[[9, 9], [8, 8]]]})
    assert json.loads((tmp_path / GEOMETRY_NAME).read_text())["lanes"][0][0] == [9, 9]


def test_empty_geometry_is_not_written(tmp_path):
    pub = TrafficViewPublisher(tmp_path, interval=0.05)
    pub.publish_geometry(None)
    pub.publish_geometry({"lanes": []})
    assert not (tmp_path / GEOMETRY_NAME).exists()
    assert not pub.geometry_written


def _run_with_view(root, run_id="pcmappo/run_1"):
    run_dir = root / run_id
    store = MetricsStore(run_dir)
    store.log_scalar("Episode/reward", 1.0, step=0)
    store.close()
    pub = TrafficViewPublisher(run_dir, interval=0.05)
    pub.publish_geometry({"lanes": [[[0, 0], [10, 0]]], "junction": None, "y_up": True})
    pub.publish_state(_vehicles(), episode=1, step=42)
    return run_dir


def test_monitor_serves_traffic_routes(tmp_path):
    _run_with_view(tmp_path)
    app = MonitorApp(tmp_path)
    geometry = app.handle("/api/runs/pcmappo/run_1/traffic_geometry", {})
    assert geometry["lanes"] == [[[0, 0], [10, 0]]]
    state = app.handle("/api/runs/pcmappo/run_1/traffic_state", {})
    assert state["step"] == 42 and len(state["vehicles"]) == 2


def test_monitor_404s_without_view_files(tmp_path):
    run_dir = tmp_path / "sac" / "bare"
    store = MetricsStore(run_dir)
    store.close()
    app = MonitorApp(tmp_path)
    with pytest.raises(FileNotFoundError):
        app.handle("/api/runs/sac/bare/traffic_state", {})


def test_monitor_traffic_route_stays_inside_root(tmp_path):
    _run_with_view(tmp_path)
    app = MonitorApp(tmp_path)
    with pytest.raises(FileNotFoundError):
        app.handle("/api/runs/../../etc/traffic_state", {})
