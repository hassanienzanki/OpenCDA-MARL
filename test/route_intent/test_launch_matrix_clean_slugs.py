"""--clean-planned-slugs removes every artifact of a planned slug, and only that slug.

The monitor's metrics store is keyed by the slug as well
(``<root>/metrics/<algorithm>/<slug>/metrics.sqlite``). Left behind, a
relaunched cell reopens it and appends the new run's steps and episodes to
the old run's, and a page's ``step > since`` cursor shows nothing new until
the new run passes the old maximum.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts" / "experiments"


def _load_launcher():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import importlib
    return importlib.import_module("launch_matrix")


def _point_roots_at(lm, monkeypatch, root: Path) -> None:
    monkeypatch.setattr(lm, "LOGS_ROOT", root / "logs")
    monkeypatch.setattr(lm, "CHECKPOINTS_MATRIX_ROOT", root / "checkpoints" / "matrix")
    monkeypatch.setattr(lm, "METRICS_MATRIX_ROOT", root / "metrics_history" / "matrix")
    monkeypatch.setattr(lm, "METRICS_STORE_ROOT", root / "metrics")
    monkeypatch.setattr(lm, "TRACES_ROOT", root / "traces")
    monkeypatch.setattr(lm, "RUNTIME_CONFIG_ROOT", root / "_runtime" / "momarl" / "intersection")


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    return path


def test_the_metrics_store_of_the_slug_is_removed_with_the_rest(tmp_path, monkeypatch):
    lm = _load_launcher()
    _point_roots_at(lm, monkeypatch, tmp_path)
    slug, other = "t_custom_learn_path_pcmappo_aligned_seed0", "t_custom_learn_path_pcmappo_aligned_seed1"

    planned = [
        _touch(tmp_path / "logs" / slug / "run.log"),
        _touch(tmp_path / "checkpoints" / "matrix" / slug / "latest_checkpoint.pth"),
        _touch(tmp_path / "metrics_history" / "matrix" / slug / "metrics_history_ep1.pkl"),
        _touch(tmp_path / "traces" / slug / "trace.jsonl"),
        _touch(tmp_path / "metrics" / "pcmappo" / slug / "metrics.sqlite"),
        _touch(tmp_path / "metrics" / "pcmappo" / slug / "traffic_1.json"),
        _touch(tmp_path / "metrics" / "masac" / slug / "metrics.sqlite"),
        _touch(tmp_path / "_runtime" / "momarl" / "intersection" / f"{slug}.yaml"),
    ]
    kept = [
        _touch(tmp_path / "metrics" / "pcmappo" / other / "metrics.sqlite"),
        _touch(tmp_path / "metrics_history" / "matrix" / other / "metrics_history_ep1.pkl"),
        _touch(tmp_path / "metrics" / "pcmappo" / "notes.txt"),
    ]

    lm._clean_cell_artifacts(slug)

    assert not any(p.exists() for p in planned), [str(p) for p in planned if p.exists()]
    assert all(p.exists() for p in kept), "another slug's artifacts must survive"
    assert (tmp_path / "metrics" / "pcmappo").is_dir(), "the algorithm directory itself stays"


def test_a_slug_is_matched_literally_not_as_a_pattern(tmp_path, monkeypatch):
    """A --slug-prefix may contain glob metacharacters; ``[ab]`` must not
    select the store of a slug spelled with ``a``."""
    lm = _load_launcher()
    _point_roots_at(lm, monkeypatch, tmp_path)
    store_a = _touch(tmp_path / "metrics" / "pcmappo" / "pa_custom_x_seed0" / "metrics.sqlite")
    literal = _touch(tmp_path / "metrics" / "pcmappo" / "p[ab]_custom_x_seed0" / "metrics.sqlite")

    lm._clean_cell_artifacts("p[ab]_custom_x_seed0")

    assert not literal.exists()
    assert store_a.exists()


def test_cleaning_with_no_metrics_store_is_a_no_op(tmp_path, monkeypatch):
    lm = _load_launcher()
    _point_roots_at(lm, monkeypatch, tmp_path)
    lm._clean_cell_artifacts("t_custom_speed_only_pcmappo_aligned_seed0")   # nothing exists yet
