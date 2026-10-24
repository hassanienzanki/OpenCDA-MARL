"""Slug prefixes and default output locations of the experiment scripts.

The matrix launchers name every cell ``<prefix><map>_<condition>_<regime>_seed<N>``
with ``--slug-prefix`` (default ``matrix_``). Cells of a run launched under
another prefix stay reachable: the continuation cells look their source up
under ``--continue-source-prefix``, which defaults to the run's own slug
prefix (also in the eval runner and through the distributed launcher).
"""
from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import os
import sys
from argparse import Namespace
from pathlib import Path

import pytest
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts" / "experiments"


def _mod(name):
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    return importlib.import_module(name)


def test_default_prefix_names_the_cells():
    common = _mod("matrix_common")
    assert common.DEFAULT_SLUG_PREFIX == "matrix_"
    cell = _mod("launch_matrix").Cell("custom", "learn_path_pcmappo", "aligned", 0)
    assert cell.slug() == "matrix_custom_learn_path_pcmappo_aligned_seed0"
    eval_cell = _mod("launch_eval_matrix").Cell("custom", "learn_path_pcmappo", "aligned", 0)
    assert eval_cell.eval_slug() == "matrix_custom_learn_path_pcmappo_aligned_seed0_eval200vph"


def test_schema_check_requires_an_action_space():
    common = _mod("matrix_common")
    cfg = OmegaConf.create({"MARL": {"algorithm": "pcmappo"}})
    with pytest.raises(ValueError, match="MARL.action_space is required"):
        common.validate_matrix_schema(cfg, source="unit-test")


def _continue_cfg():
    return OmegaConf.create({"MARL": {"training": {"load_checkpoint": ""}}})


def _source_checkpoint(root: Path, prefix: str) -> Path:
    slug = _mod("matrix_common").slug_for(
        map_slug="custom", condition="learn_path_pcmappo", regime="aligned",
        seed=0, prefix=prefix)
    ckpt = root / slug / "pcmappo" / "best_checkpoint.pth"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"")
    return ckpt


def test_continuation_finds_its_source_under_the_given_prefix(tmp_path):
    lm = _mod("launch_matrix")
    ckpt = _source_checkpoint(tmp_path, "earlier_tag_")
    cfg = _continue_cfg()
    lm._maybe_inject_continue_checkpoint(
        cfg, condition="continue_learn_path_pcmappo", regime="aligned", seed=0,
        continue_source_root=tmp_path, continue_source_map="custom",
        continue_source_prefix="earlier_tag_")
    assert Path(cfg.MARL.training.load_checkpoint) == ckpt


def test_continuation_defaults_to_the_default_prefix(tmp_path):
    lm = _mod("launch_matrix")
    _source_checkpoint(tmp_path, "earlier_tag_")
    with pytest.raises(FileNotFoundError, match="matrix_custom_learn_path_pcmappo"):
        lm._maybe_inject_continue_checkpoint(
            _continue_cfg(), condition="continue_learn_path_pcmappo",
            regime="aligned", seed=0, continue_source_root=tmp_path,
            continue_source_map="custom")
    ckpt = _source_checkpoint(tmp_path, "matrix_")
    cfg = _continue_cfg()
    lm._maybe_inject_continue_checkpoint(
        cfg, condition="continue_learn_path_pcmappo", regime="aligned", seed=0,
        continue_source_root=tmp_path, continue_source_map="custom")
    assert Path(cfg.MARL.training.load_checkpoint) == ckpt


def test_patch_config_forwards_the_continuation_prefix(tmp_path, monkeypatch):
    lm = _mod("launch_matrix")
    seen = {}

    def fake_inject(cfg, **kw):
        seen.update(kw)

    monkeypatch.setattr(lm, "_maybe_inject_continue_checkpoint", fake_inject)
    src = lm.CONFIG_BASE / "learn_path_pcmappo.yaml"
    lm._patch_config(src, tmp_path / "cell.yaml", "aligned", 0, "slug",
                     condition="continue_learn_path_pcmappo",
                     continue_source_prefix="earlier_tag_")
    assert seen["continue_source_prefix"] == "earlier_tag_"
    lm._patch_config(src, tmp_path / "cell.yaml", "aligned", 0, "slug",
                     condition="learn_path_pcmappo")
    assert seen["continue_source_prefix"] == lm.DEFAULT_SLUG_PREFIX


def test_eval_runner_finds_a_continuation_source_under_the_cells_prefix(tmp_path, monkeypatch):
    """run_matrix_eval regenerates a continuation cell's runtime profile from
    its slug; the source lives under that slug's own prefix, not the default."""
    lm = _mod("launch_matrix")
    rme = _mod("run_matrix_eval")
    monkeypatch.setattr(lm, "CHECKPOINTS_MATRIX_ROOT", tmp_path / "checkpoints")
    monkeypatch.setattr(rme, "RUNTIME_CONFIG_ROOT", tmp_path / "runtime")
    ckpt = _source_checkpoint(tmp_path / "checkpoints", "earlier_tag_")

    slug = "earlier_tag_custom_continue_learn_path_pcmappo_aligned_seed0"
    dst = rme.regenerate_runtime_yaml(slug, rme.parse_slug(slug), max_steps=100,
                                      active_junctions=[4], carla_rpc_port=2000)
    loaded = Path(OmegaConf.load(dst).MARL.training.load_checkpoint)
    assert loaded.parent.parent.name == "earlier_tag_custom_learn_path_pcmappo_aligned_seed0"
    assert loaded.name == ckpt.name


def test_eval_runner_keeps_the_map_profiles_junctions_by_default(tmp_path, monkeypatch):
    """The eval runner's --active-junctions defaults to None, as in the
    launchers: each cell then drives its map profile's junctions, and an
    explicit list overrides them."""
    rme = _mod("run_matrix_eval")
    slug = "matrix_custom_learn_path_pcmappo_aligned_seed0"
    ckpt = tmp_path / "checkpoints" / slug / "pcmappo" / "best_checkpoint.pth"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"")
    monkeypatch.setattr(rme, "CHECKPOINTS_ROOT", tmp_path / "checkpoints")
    monkeypatch.setattr(rme, "EVAL_OUT_ROOT", tmp_path / "eval")
    monkeypatch.setattr(rme, "SUMMARY_PATH", tmp_path / "summary.json")
    monkeypatch.setattr(rme, "RUNTIME_CONFIG_ROOT", tmp_path / "runtime")
    monkeypatch.setattr(rme, "scenario_ref_for", lambda path: "unused")
    monkeypatch.setattr(rme, "run_eval_for_cell", lambda *a, **kw: {})
    written = []
    regenerate = rme.regenerate_runtime_yaml

    def recording_regenerate(*args, **kwargs):
        dst = regenerate(*args, **kwargs)
        written.append(list(OmegaConf.load(dst).scenario.traffic.active_junctions))
        return dst

    monkeypatch.setattr(rme, "regenerate_runtime_yaml", recording_regenerate)
    # A profile whose junctions differ from any built-in constant, so the
    # default can only come from the profile.
    common = _mod("matrix_common")
    monkeypatch.setitem(common.MAP_PROFILES, "custom", dataclasses.replace(
        common.MAP_PROFILES["custom"], active_junctions=(2, 3)))
    for argv in ([], ["--active-junctions", "1"]):
        monkeypatch.setattr(sys, "argv", ["run_matrix_eval.py", *argv])
        rme.main()
    assert written == [[2, 3], [1]]


def _main_continue_prefix(monkeypatch, *argv):
    """Run launch_matrix.main as a dry run of one continuation cell and return
    the continuation prefix it hands to the cell launch."""
    lm = _mod("launch_matrix")
    seen = {}

    def fake_launch(cell, **kw):
        seen.update(kw)
        return 0

    monkeypatch.setattr(lm, "_launch_cell", fake_launch)
    monkeypatch.setattr(sys, "argv", [
        "launch_matrix.py", "--dry-run", "--conditions", "continue_learn_path_pcmappo",
        "--regimes", "aligned", "--seeds", "0", *argv])
    assert lm.main() == 0
    return seen["continue_source_prefix"]


def test_continuation_prefix_follows_the_slug_prefix(monkeypatch):
    assert _main_continue_prefix(monkeypatch, "--slug-prefix", "earlier_tag_") == "earlier_tag_"
    assert _main_continue_prefix(monkeypatch) == "matrix_"
    assert _main_continue_prefix(monkeypatch, "--slug-prefix", "") == ""
    # An explicit source prefix wins; legacy runs keep the default lookup.
    assert _main_continue_prefix(
        monkeypatch, "--slug-prefix", "new_", "--continue-source-prefix", "earlier_tag_",
    ) == "earlier_tag_"
    assert _main_continue_prefix(monkeypatch, "--legacy-slugs") == "matrix_"


def test_distributed_launcher_forwards_the_continuation_prefix(monkeypatch):
    dist = _mod("launch_distributed_matrix")
    commands = []

    class _Done:
        returncode = 0

    def fake_run(cmd, cwd=None):
        commands.append(cmd)
        return _Done()

    monkeypatch.setattr(dist.subprocess, "run", fake_run)
    worker = dist.Worker(name="w", os_name="linux", gpu_id="0", carla_rpc_port=2000,
                         carla_streaming_port=0, max_concurrent_cells=1,
                         artifact_root=REPO)
    cell = dist.Cell("custom", "continue_learn_path_pcmappo", "aligned", 0)
    base = dict(slug_prefix="new_", carla_restart_mode="none", resume=False,
                max_episodes=None, max_steps=None, cell_timeout=None,
                active_junctions=None, dry_run=False)

    assert dist._launch_worker_cells(
        worker, [cell], Namespace(**base, continue_source_prefix="earlier_tag_")) == 0
    cmd = commands[-1]
    assert cmd[cmd.index("--continue-source-prefix") + 1] == "earlier_tag_"
    assert cmd[cmd.index("--slug-prefix") + 1] == "new_"

    dist._launch_worker_cells(worker, [cell], Namespace(**base, continue_source_prefix=None))
    assert "--continue-source-prefix" not in commands[-1]


def _load_checkpoint_eval():
    """Import scripts/run_carla_checkpoint_eval.py without leaking the
    environment defaults it sets at import time into other tests."""
    added = [k for k in ("KMP_DUPLICATE_LIB_OK", "LOGURU_LEVEL") if k not in os.environ]
    try:
        spec = importlib.util.spec_from_file_location(
            "run_carla_checkpoint_eval_under_test",
            REPO / "scripts" / "run_carla_checkpoint_eval.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for key in added:
            os.environ.pop(key, None)


def test_checkpoint_eval_writes_under_eval_runs_by_default(tmp_path):
    m = _load_checkpoint_eval()
    args = Namespace(output="", scenario="momarl/intersection/learn_path_pcmappo",
                     seed=3, episodes=5)
    out = m.resolve_output_path(args, Path("best_checkpoint.pth"))
    assert out.name == "eval_summary.json"
    assert out.parent.parent == REPO / "artifacts" / "eval_runs"
    explicit = tmp_path / "summary.json"
    args.output = str(explicit)
    assert m.resolve_output_path(args, Path("best_checkpoint.pth")) == explicit
