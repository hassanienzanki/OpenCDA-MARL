"""launch_matrix.py --parallel: per-slot ports, per-slot GPUs, port-scoped CARLA restarts.

Every slot owns one CARLA server (rpc base + 10*k). A slot's restart must
touch only its own server, and with several GPUs listed the learners and
the CARLA servers are spread round-robin over the cards.
"""
from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts" / "experiments"


def _load_launcher():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import importlib
    return importlib.import_module("launch_matrix")


def test_round_robin_spreads_lists_and_passes_scalars_through():
    lm = _load_launcher()
    assert lm._round_robin(None, 3) is None
    assert lm._round_robin("", 3) is None
    assert lm._round_robin("0", 5) == "0"
    assert [lm._round_robin("0,1", k) for k in range(4)] == ["0", "1", "0", "1"]
    assert [lm._round_robin(" 2, 5 ,7", k) for k in range(4)] == ["2", "5", "7", "2"]


def test_parallel_slots_get_their_own_port_gpu_and_restart(monkeypatch):
    lm = _load_launcher()
    launched, restarted = [], []

    class _Cell:
        def __init__(self, i):
            self.i = i

        def slug(self, prefix, legacy=False):
            return f"{prefix}_cell{self.i}"

    def fake_launch(cell, **kw):
        launched.append((cell.i, kw["carla_rpc_port"], kw["cuda_visible_devices"]))
        return 0

    def fake_restart(**kw):
        restarted.append((kw["rpc_port"], kw.get("graphics_adapter")))
        return True

    monkeypatch.setattr(lm, "_launch_cell", fake_launch)
    monkeypatch.setattr(lm, "_restart_carla", fake_restart)
    monkeypatch.setattr(lm, "_carla_responsive", lambda **kw: False)   # every slot restarts once
    monkeypatch.setattr(lm, "_cell_done", lambda *a, **k: False)

    args = Namespace(
        parallel=3, carla_rpc_port=2000, carla_streaming_port=0,
        carla_restart_mode="process", carla_docker_name="x", render=False,
        cuda_visible_devices="0,1", carla_gpus="1,0",
        dry_run=False, resume=False, legacy_slugs=False,
        max_episodes=None, max_steps=None, cell_timeout=None,
        active_junctions=None, enable_trace=False, trace_sample_every=1,
        continue_source_root=None, continue_source_map=None,
    )
    lm._run_parallel([_Cell(i) for i in range(6)], args, slug_prefix="t",
                     carla_bin=Path("nonexistent"), resume_min=1)

    assert len(launched) == 6
    ports = {p for _, p, _ in launched}
    assert ports <= {2000, 2010, 2020}
    # A learner on rpc 2000/2020 (slots 0/2) sees GPU 0, on 2010 (slot 1) GPU 1.
    for _, port, gpu in launched:
        assert gpu == ("1" if port == 2010 else "0"), (port, gpu)
    # Each slot restarted only its own server, on its own CARLA GPU
    # (--carla-gpus 1,0 -> slot 0/2 on GPU 1, slot 1 on GPU 0). Which slots
    # got cells depends on scheduling, so check the mapping for those seen.
    by_port = {p: g for p, g in restarted}
    assert set(by_port) == ports
    expected_carla_gpu = {2000: 1, 2010: 0, 2020: 1}
    for port, gpu in by_port.items():
        assert gpu == expected_carla_gpu[port], (port, gpu)


def test_kill_carla_only_touches_owned_or_matching_processes(monkeypatch):
    """_kill_carla kills the owned handle by tree and searches by the
    -carla-rpc-port marker; it never issues an image-wide kill."""
    lm = _load_launcher()
    calls = []

    class _Proc:
        pid = 4242

        def poll(self):
            return None

    def fake_run(cmd, **kw):
        calls.append(cmd)

        class R:
            returncode = 0
        return R()

    monkeypatch.setattr(lm.subprocess, "run", fake_run)
    monkeypatch.setattr(lm.os, "name", "nt")
    lm._OWNED_CARLA[2010] = _Proc()
    lm._kill_carla(2010)
    assert 2010 not in lm._OWNED_CARLA
    flat = [" ".join(map(str, c)) for c in calls]
    assert any("/PID 4242" in c and "/T" in c for c in flat), flat
    assert any("-carla-rpc-port=2010" in c for c in flat), flat
    assert not any("/IM" in c for c in flat), "must not kill every CARLA image on the host"


def test_cuda_graph_update_flag_lands_in_the_cell_config(tmp_path):
    lm = _load_launcher()
    src = tmp_path / "src.yaml"
    yaml_lines = [
        "MARL:",
        "  algorithm: pcmasac",
        "  action_space:",
        "    mode: speed_only",
        "  reward_profile: simple_progress_v1",
        "  state_dim: 8",
        "  action_dim: 1",
        "  pcmasac:",
        "    batch_size: 4",
        "  training: {}",
        "outputs: {}",
        "scenario:",
        "  simulation: {}",
        "agents: {}",
        "world: {}",
    ]
    src.write_text(chr(10).join(yaml_lines) + chr(10), encoding="utf-8")
    dst = tmp_path / "cell.yaml"
    lm._patch_config(src, dst, "aligned", 0, "slug", condition="speed_only_pcmappo",
                     cuda_graph_update=True)
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(dst)
    assert cfg.MARL.pcmasac.cuda_graph_update is True
    assert cfg.MARL.pcmasac.batch_size == 4
    lm._patch_config(src, dst, "aligned", 0, "slug", condition="speed_only_pcmappo")
    assert "cuda_graph_update" not in OmegaConf.load(dst).MARL.pcmasac   # untouched by default


def test_restart_carla_drops_sdl_videodriver_from_the_server_env(monkeypatch, tmp_path):
    """UE4's SDL2 exits at once under SDL_VIDEODRIVER=offscreen (the pixi Linux
    activation sets it for pygame); the spawned server must not inherit it."""
    lm = _load_launcher()
    spawned = {}

    class _Proc:
        pid = 1
        def poll(self):
            return None

    def fake_popen(args, **kw):
        spawned["args"], spawned["kw"] = args, kw
        return _Proc()

    monkeypatch.setenv("SDL_VIDEODRIVER", "offscreen")
    monkeypatch.setenv("CARLA_SMOKE_MARKER", "keep-me")
    monkeypatch.setattr(lm.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(lm, "_kill_carla", lambda port: None)
    monkeypatch.setattr(lm, "_carla_responsive", lambda **kw: True)
    monkeypatch.setattr(lm.time, "sleep", lambda s: None)
    carla_bin = tmp_path / "CarlaUE4.sh"
    carla_bin.write_text("#!/bin/sh")
    assert lm._restart_carla(carla_bin=carla_bin, rpc_port=2010, streaming_port=0,
                             mode="process", docker_name="x", graphics_adapter=1) is True
    env = spawned["kw"]["env"]
    assert "SDL_VIDEODRIVER" not in env
    assert env["CARLA_SMOKE_MARKER"] == "keep-me"
    assert "-graphicsadapter=1" in spawned["args"] and "-carla-rpc-port=2010" in spawned["args"]


def test_stop_owned_carla_only_stops_servers_the_launcher_started(monkeypatch):
    lm = _load_launcher()
    killed = []
    monkeypatch.setattr(lm, "_kill_carla", lambda port: killed.append(port))
    lm._OWNED_CARLA.clear()
    lm._OWNED_CARLA[2010] = object()
    lm._OWNED_CARLA[2000] = object()
    lm._stop_owned_carla()
    assert killed == [2000, 2010]          # owned ports only, deterministic order
    lm._OWNED_CARLA.clear()
    lm._stop_owned_carla()
    assert killed == [2000, 2010]          # nothing owned -> nothing touched

def test_default_conditions_are_real_conditions():
    """A bare `launch_matrix.py` must plan: the old default named
    'speed_only_pcma', which has not existed since the pcma/momarl split, so
    the parser default raised ValueError: unknown condition before anything
    launched."""
    import argparse

    lm = _load_launcher()
    parser = argparse.ArgumentParser()
    defaults = None
    src = (SCRIPTS / "launch_matrix.py").read_text(encoding="utf-8")
    marker = '--conditions", nargs="+",'
    assert marker in src
    tail = src.split(marker, 1)[1]
    default_line = tail.split("default=[", 1)[1].split("]", 1)[0]
    defaults = [d.strip().strip('"').strip("'") for d in default_line.split(",") if d.strip()]
    del parser
    assert defaults, "no default conditions parsed"
    for name in defaults:
        assert name in lm.CONDITION_CONFIGS, f"default condition {name!r} does not exist"


def _args(**overrides):
    base = dict(parallel=1, carla_rpc_port=2000, carla_streaming_port=0,
                carla_docker_name="opencda-carla-2000", carla_restart_mode="none",
                cuda_visible_devices=None, carla_gpus=None, resume=False,
                dry_run=False, render=False, legacy_slugs=False,
                max_episodes=1, max_steps=10, cell_timeout=None,
                active_junctions=None, enable_trace=False, trace_sample_every=1,
                continue_source_root=None, continue_source_map=None,
                cuda_graph_update=None)
    base.update(overrides)
    return Namespace(**base)


def _cell(lm):
    return lm.Cell(map_slug="custom", condition="speed_only_pcmappo",
                   regime="aligned", seed=41)


def test_slot_failure_is_counted_and_returned(monkeypatch, tmp_path):
    """A slot that raises must mark its cell failed; _run_parallel returns the
    count so main() can exit non-zero. Before, the thread died silently and the
    summary printed '0 cell(s), 0 failed' with exit status 0."""
    lm = _load_launcher()

    def boom(*a, **kw):
        raise RuntimeError("bad config")

    monkeypatch.setattr(lm, "_patch_config", boom)
    monkeypatch.setattr(lm, "_carla_responsive", lambda **kw: True)
    failed = lm._run_parallel([_cell(lm)], _args(), slug_prefix="t_",
                              carla_bin=tmp_path / "x", resume_min=1)
    assert failed == 1


def test_docker_slot_uses_the_port_named_container(monkeypatch, tmp_path):
    """docker_carla.py names containers opencda-carla-<rpc-port>; the slot must
    restart that one (the old '<base>-<slot>' name existed for no container)."""
    lm = _load_launcher()
    seen = []
    monkeypatch.setattr(lm, "_carla_responsive", lambda **kw: False)
    monkeypatch.setattr(lm, "_patch_config", lambda *a, **kw: tmp_path / "c.yaml")
    monkeypatch.setattr(lm, "_launch_cell", lambda *a, **kw: 0)

    def fake_restart(**kw):
        seen.append(kw["docker_name"])
        return True

    monkeypatch.setattr(lm, "_restart_carla", fake_restart)
    lm._run_parallel([_cell(lm)], _args(carla_rpc_port=2010, carla_restart_mode="docker"),
                     slug_prefix="t_", carla_bin=tmp_path / "x", resume_min=1)
    assert seen == ["opencda-carla-2010"], seen


def test_posix_kill_pattern_is_anchored_on_the_carla_binary(monkeypatch):
    """`pkill -f -- -carla-rpc-port=2000` also matched the launcher's own argv
    and any server on port 20000+."""
    lm = _load_launcher()
    calls = []
    monkeypatch.setattr(lm.os, "name", "posix")
    monkeypatch.setattr(lm.subprocess, "run",
                        lambda cmd, **kw: calls.append(cmd) or type("R", (), {"returncode": 0}))
    lm._OWNED_CARLA.clear()
    lm._kill_carla(2000)
    pattern = calls[0][-1]
    assert pattern.startswith("CarlaUE4"), pattern
    import re
    assert re.search(pattern, "CarlaUE4-Linux-Shipping CarlaUE4 -carla-rpc-port=2000 -x")
    assert not re.search(pattern, "python launch_matrix.py --carla-rpc-port=2000")
    assert not re.search(pattern, "CarlaUE4-Linux-Shipping -carla-rpc-port=20000")
