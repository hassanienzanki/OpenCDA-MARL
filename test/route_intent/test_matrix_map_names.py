"""Map arguments of the matrix launchers name cells by their profile key.

The launchers accept a map case-insensitively (``--maps Custom``), but
run_matrix_eval and every analyzer match the map in a slug exactly against
the ``MAP_PROFILES`` keys. Each launcher therefore carries the canonical key
from validation on, so ``--maps Custom`` names ``matrix_custom_*`` cells
rather than ``matrix_Custom_*`` ones nothing downstream reads. Spellings
of one map plan its cells once, and an unknown map still fails before
anything is planned, with the same message.
"""
from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts" / "experiments"


def _mod(name):
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    return importlib.import_module(name)


def _unknown_map_message(name: str) -> str:
    profiles = sorted(_mod("matrix_common").MAP_PROFILES)
    return re.escape(f"unknown map profile {name!r}; valid: {profiles}")


def _write_manifest(tmp_path):
    manifest = tmp_path / "workers.yaml"
    manifest.write_text(
        "workers:\n"
        "  - name: w\n"
        "    gpu_id: 0\n"
        "    carla_rpc_port: 2000\n"
        f"    artifact_root: {tmp_path.as_posix()}\n",
        encoding="utf-8",
    )
    return manifest


def test_canonical_map_is_the_profile_key():
    common = _mod("matrix_common")
    for name in ("custom", "Custom", "CUSTOM"):
        assert common.canonical_map(name) == "custom"
        assert common.map_profile(name) is common.MAP_PROFILES["custom"]
    with pytest.raises(ValueError, match=_unknown_map_message("Nowhere")):
        common.canonical_map("Nowhere")
    with pytest.raises(ValueError, match=_unknown_map_message("Nowhere")):
        common.map_profile("Nowhere")


def test_canonical_maps_keeps_each_profile_once_in_order(monkeypatch):
    common = _mod("matrix_common")
    monkeypatch.setitem(common.MAP_PROFILES, "other", common.MAP_PROFILES["custom"])
    assert common.canonical_maps(["Other", "custom", "OTHER", "Custom"]) == [
        "other", "custom"]
    with pytest.raises(ValueError, match=_unknown_map_message("Nowhere")):
        common.canonical_maps(["custom", "Nowhere"])


def test_launch_matrix_names_cells_by_the_profile_key(monkeypatch, capsys):
    lm = _mod("launch_matrix")
    launched = []
    launch = lm._launch_cell

    def recording_launch(cell, **kw):
        launched.append((cell, kw))
        return launch(cell, **kw)

    monkeypatch.setattr(lm, "_launch_cell", recording_launch)
    monkeypatch.setattr(sys, "argv", [
        "launch_matrix.py", "--dry-run", "--maps", "Custom",
        "--conditions", "learn_path_pcmappo", "continue_learn_path_pcmappo",
        "--regimes", "aligned", "--seeds", "0",
        "--continue-source-map", "CUSTOM"])
    assert lm.main() == 0

    assert [cell.slug() for cell, _ in launched] == [
        "matrix_custom_learn_path_pcmappo_aligned_seed0",
        "matrix_custom_continue_learn_path_pcmappo_aligned_seed0",
    ]
    assert {kw["continue_source_map"] for _, kw in launched} == {"custom"}
    out = capsys.readouterr().out
    assert "[matrix] maps: custom\n" in out
    assert "[matrix] launching matrix_custom_learn_path_pcmappo_aligned_seed0 (custom)" in out
    assert "matrix_Custom" not in out


def test_launch_eval_matrix_names_cells_by_the_profile_key(tmp_path, monkeypatch, capsys):
    lem = _mod("launch_eval_matrix")
    ckpt = (tmp_path / "matrix_custom_learn_path_pcmappo_aligned_seed0"
            / "pcmappo" / "best_checkpoint.pth")
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"")
    launched = []
    launch = lem._launch_eval

    def recording_launch(cell, *args, **kw):
        launched.append((cell, kw))
        return launch(cell, *args, **kw)

    monkeypatch.setattr(lem, "_launch_eval", recording_launch)
    monkeypatch.setattr(sys, "argv", [
        "launch_eval_matrix.py", "--dry-run", "--maps", "Custom",
        "--checkpoint-map", "CUSTOM", "--checkpoint-root", str(tmp_path),
        "--conditions", "learn_path_pcmappo", "--regimes", "aligned", "--seeds", "0"])
    lem.main()

    [(cell, kw)] = launched
    assert cell.eval_slug() == "matrix_custom_learn_path_pcmappo_aligned_seed0_eval200vph"
    assert kw["checkpoint_map"] == "custom"
    out = capsys.readouterr().out
    assert "[eval] maps: custom\n" in out
    assert ("[eval] launching matrix_custom_learn_path_pcmappo_aligned_seed0_eval200vph "
            "(custom)") in out
    assert "matrix_Custom" not in out


def test_distributed_launcher_names_and_forwards_the_profile_key(tmp_path, monkeypatch, capsys):
    dist = _mod("launch_distributed_matrix")
    manifest = _write_manifest(tmp_path)

    def no_run(*args, **kwargs):
        raise AssertionError("a dry run starts no cell")

    monkeypatch.setattr(dist.subprocess, "run", no_run)
    monkeypatch.setattr(sys, "argv", [
        "launch_distributed_matrix.py", "--manifest", str(manifest), "--worker", "w",
        "--dry-run", "--maps", "Custom", "--conditions", "learn_path_pcmappo",
        "--regimes", "aligned", "--seeds", "0", "--carla-restart-mode", "none"])
    with pytest.raises(SystemExit) as done:
        dist.main()
    assert done.value.code == 0

    out = capsys.readouterr().out
    assert "[dist]   matrix_custom_learn_path_pcmappo_aligned_seed0\n" in out
    assert "[dist] w: launching matrix_custom_learn_path_pcmappo_aligned_seed0\n" in out
    assert " --maps custom " in out
    assert "--maps Custom" not in out


@pytest.mark.parametrize("script, planned", [
    ("launch_matrix", "[matrix] planning 1 cells\n"),
    ("launch_eval_matrix", "[eval] planning 1 eval cells\n"),
    ("launch_distributed_matrix", "[dist] planned cells: 1\n"),
])
def test_spellings_of_one_map_plan_its_cells_once(script, planned, tmp_path,
                                                   monkeypatch, capsys):
    module = _mod(script)
    extra = {
        "launch_matrix": [],
        "launch_eval_matrix": ["--checkpoint-root", str(tmp_path)],
        "launch_distributed_matrix": [
            "--manifest", str(_write_manifest(tmp_path)), "--worker", "w",
            "--carla-restart-mode", "none"],
    }[script]
    if script == "launch_distributed_matrix":
        def no_run(*args, **kwargs):
            raise AssertionError("a dry run starts no cell")
        monkeypatch.setattr(module.subprocess, "run", no_run)
    monkeypatch.setattr(sys, "argv", [
        f"{script}.py", "--dry-run", "--maps", "custom", "Custom",
        "--conditions", "learn_path_pcmappo", "--regimes", "aligned",
        "--seeds", "0", *extra])
    try:
        module.main()
    except SystemExit as done:
        assert done.code in (None, 0)

    out = capsys.readouterr().out
    assert planned in out
    assert "matrix_Custom" not in out


@pytest.mark.parametrize("script, argv", [
    ("launch_matrix", ["--maps", "Nowhere"]),
    ("launch_matrix", ["--continue-source-map", "Nowhere"]),
    ("launch_eval_matrix", ["--maps", "Nowhere"]),
    ("launch_eval_matrix", ["--checkpoint-map", "Nowhere"]),
    ("launch_distributed_matrix", ["--maps", "Nowhere"]),
])
def test_an_unknown_map_fails_before_planning(script, argv, tmp_path, monkeypatch, capsys):
    module = _mod(script)
    extra = ["--checkpoint-root", str(tmp_path)] if script == "launch_eval_matrix" else []
    monkeypatch.setattr(sys, "argv", [f"{script}.py", "--dry-run", *argv, *extra])
    with pytest.raises(ValueError, match=_unknown_map_message("Nowhere")):
        module.main()
    assert capsys.readouterr().out == ""
