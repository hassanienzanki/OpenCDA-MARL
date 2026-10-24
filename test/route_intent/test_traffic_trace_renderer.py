import json
from pathlib import Path

import numpy as np

from opencda_marl.telemetry.trace import RouteActionTraceWriter
from scripts.experiments.render_traffic_trace import render_frame


def test_render_traffic_trace_frame(tmp_path: Path):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "routes.json").write_text(
        json.dumps([
            {
                "route_id": 7,
                "junction_id": 4,
                "route_type": "left",
                "entry_direction": "north",
                "exit_direction": "west",
                "entry_lane_idx": 0,
                "exit_lane_idx": 1,
                "spawn_tf": {"location": {"x": 0.0, "y": 0.0, "z": 0.0}},
                "dest_tf": {"location": {"x": -10.0, "y": 10.0, "z": 0.0}},
            }
        ]),
        encoding="utf-8",
    )
    (trace_dir / "steps.jsonl").write_text(
        json.dumps({
            "type": "step",
            "episode": 0,
            "step": 12,
            "events": [],
            "agents": [
                {
                    "agent_id": "42",
                    "x": 1.0,
                    "y": 2.0,
                    "heading": 0.0,
                    "speed": 5.0,
                    "target_speed": 19.0,
                    "preference": [0.75, 0.25],
                    "route_type": "left",
                    "route_intent": {
                        "selected_intent": "left",
                        "selected_route_id": 7,
                    },
                }
            ],
        }) + "\n",
        encoding="utf-8",
    )

    out = render_frame(trace_dir, tmp_path / "frame.png")

    assert out.exists()
    assert out.stat().st_size > 0


def test_trace_agent_record_accepts_numpy_preference(tmp_path: Path):
    writer = RouteActionTraceWriter({"enabled": False, "root_dir": str(tmp_path)}, None)

    record = writer._agent_record(
        42,
        {
            "position_x": 1.0,
            "position_y": 2.0,
            "preference": np.asarray([0.75, 0.25], dtype=np.float32),
            "route_intent": {"selected_intent": "left"},
        },
    )

    assert record["preference"] == [0.75, 0.25]
    assert record["route_intent"]["selected_intent"] == "left"


def test_trace_retains_the_agent_reported_safety_reason(tmp_path: Path):
    writer = RouteActionTraceWriter({"enabled": False, "root_dir": str(tmp_path)}, None)
    record = writer._agent_record(42, {
        "target_speed": 30.0, "applied_speed": 0.0,
        "shield_active": True, "shield_stage": np.int64(2),
        "lane_hold": np.bool_(True),
        "merge_priority_enabled": np.bool_(True),
        "merge_priority_release": np.bool_(True),
        "merge_priority_override": np.bool_(False),
    })
    decoded = json.loads(json.dumps(record))
    assert decoded["applied_speed"] == 0.0
    assert decoded["shield_stage"] == 2
    assert decoded["lane_hold"] is True
    assert decoded["merge_priority_enabled"] is True
    assert decoded["merge_priority_release"] is True
    assert decoded["merge_priority_override"] is False
    unknown = writer._agent_record(43, {})
    assert unknown["shield_stage"] is None and unknown["lane_hold"] is None
    assert unknown["merge_priority_enabled"] is None
    assert unknown["merge_priority_release"] is None
    assert unknown["merge_priority_override"] is None
