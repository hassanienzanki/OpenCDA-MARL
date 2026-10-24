"""Aggregation helper tests."""
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from opencda_marl.evaluation.aggregate import (
    bootstrap_hv_ci,
    median_iqr,
    paired_wilcoxon,
    summarize_cell,
)


def test_median_iqr_returns_expected_values():
    out = median_iqr([1.0, 2.0, 3.0, 4.0, 5.0])
    assert out["median"] == 3.0
    assert out["q1"] == 2.0
    assert out["q3"] == 4.0
    assert out["min"] == 1.0
    assert out["max"] == 5.0


def test_median_iqr_empty_returns_nans():
    out = median_iqr([])
    assert all(np.isnan(v) for v in out.values())


def test_paired_wilcoxon_detects_systematic_difference():
    baseline = [0.50, 0.55, 0.52, 0.48, 0.51]
    treatment = [0.65, 0.70, 0.68, 0.62, 0.69]
    p_value = paired_wilcoxon(baseline, treatment)
    assert 0.0 < p_value < 0.1, f"expected small p; got {p_value}"


def test_paired_wilcoxon_small_n_returns_nan():
    p = paired_wilcoxon([0.5, 0.6], [0.7, 0.8])
    assert np.isnan(p)


def test_paired_wilcoxon_identical_returns_one():
    p = paired_wilcoxon([0.5, 0.6, 0.7], [0.5, 0.6, 0.7])
    assert p == 1.0


def test_paired_wilcoxon_shape_mismatch_raises():
    with pytest.raises(ValueError):
        paired_wilcoxon([0.5, 0.6], [0.5, 0.6, 0.7])


def test_bootstrap_hv_ci_returns_lower_upper_bounds():
    rng = np.random.default_rng(7)
    baseline_pts = rng.uniform(0.1, 0.5, size=(20, 2))
    treatment_pts = baseline_pts + 0.1
    lo, hi = bootstrap_hv_ci(baseline_pts, treatment_pts, reference=(0.0, 0.0),
                              n_bootstrap=200, rng=rng)
    assert lo <= hi
    # treatment > baseline by construction, so CI should be > 0 most of the time
    assert hi > 0


def test_summarize_cell_aggregates_across_archives(tmp_path: Path):
    # Create two synthetic seed archives.
    seed0 = tmp_path / "seed0"
    seed1 = tmp_path / "seed1"
    seed0.mkdir()
    seed1.mkdir()
    (seed0 / "ep0.json").write_text(json.dumps({"collision_rate": 0.10, "success_rate": 0.80}))
    (seed0 / "ep1.json").write_text(json.dumps({"collision_rate": 0.12, "success_rate": 0.78}))
    (seed1 / "ep0.json").write_text(json.dumps({"collision_rate": 0.08, "success_rate": 0.82}))

    summary = summarize_cell([seed0, seed1])
    assert "collision_rate" in summary
    assert summary["collision_rate"]["median"] == 0.10
    assert summary["success_rate"]["max"] == 0.82


def test_summarize_cell_ignores_missing_archives(tmp_path):
    out = summarize_cell([tmp_path / "does_not_exist"])
    assert out == {}


def test_summarize_cell_ignores_non_numeric_fields(tmp_path: Path):
    seed = tmp_path / "seed0"
    seed.mkdir()
    (seed / "ep0.json").write_text(json.dumps({"collision_rate": 0.10, "label": "foo", "ok": True}))
    summary = summarize_cell([seed])
    assert set(summary.keys()) == {"collision_rate"}


def test_summarize_cell_reads_training_metadata_history(tmp_path: Path):
    """Verify the checkpoint-style training_metadata.json layout works."""
    seed = tmp_path / "checkpoint_run"
    inner = seed / "pcmappo"
    inner.mkdir(parents=True)
    metadata = {
        "algorithm": "pcmappo",
        "best_reward": -1000.0,
        "history": [
            {"episode": 0, "metrics": {"collision_rate": 0.20, "success_rate": 0.60}},
            {"episode": 1, "metrics": {"collision_rate": 0.15, "success_rate": 0.65}},
            {"episode": 2, "metrics": {"collision_rate": 0.10, "success_rate": 0.70}},
        ],
    }
    (inner / "training_metadata.json").write_text(json.dumps(metadata))
    summary = summarize_cell([seed])
    assert "collision_rate" in summary
    assert summary["collision_rate"]["median"] == 0.15
    assert summary["success_rate"]["median"] == 0.65


def test_summarize_cell_prefers_flat_records_when_present(tmp_path: Path):
    """If both flat .json and training_metadata.json exist, flat wins."""
    seed = tmp_path / "mixed"
    seed.mkdir()
    (seed / "ep0.json").write_text(json.dumps({"collision_rate": 0.99}))
    # Add a phantom training_metadata that disagrees -- aggregator should
    # not use it because the flat records are present and non-empty.
    inner = seed / "pcmappo"
    inner.mkdir()
    (inner / "training_metadata.json").write_text(json.dumps({
        "history": [{"episode": 0, "metrics": {"collision_rate": 0.01}}]
    }))
    summary = summarize_cell([seed])
    assert summary["collision_rate"]["median"] == 0.99


def test_summarize_cell_records_from_checkpoint_layout_only(tmp_path: Path):
    """When only training_metadata is present, it is read."""
    seed = tmp_path / "checkpoint_only"
    inner = seed / "pcmappo"
    inner.mkdir(parents=True)
    metadata = {
        "history": [
            {"episode": 0, "metrics": {"reward": 100.0}},
            {"episode": 1, "metrics": {"reward": 200.0}},
            {"episode": 2, "metrics": {"reward": 300.0}},
        ],
    }
    (inner / "training_metadata.json").write_text(json.dumps(metadata))
    summary = summarize_cell([seed])
    assert summary["reward"]["median"] == 200.0
    assert summary["reward"]["min"] == 100.0
    assert summary["reward"]["max"] == 300.0


def test_summarize_cell_reads_metrics_history_pickle_archive(tmp_path: Path):
    seed = tmp_path / "eval_archive"
    seed.mkdir()
    history = {
        "rewards": [10.0, 20.0],
        "utilities": [11.0, 22.0],
        "objective_component_totals": [
            {"efficiency": 1.0, "interaction_risk": -0.2},
            {"efficiency": 2.0, "interaction_risk": -0.1},
        ],
        "objective_specs": [
            ["efficiency", "interaction_risk"],
            ["efficiency", "interaction_risk"],
        ],
        "episode_lengths": [100, 200],
        "success_counts": [1, 3],
        "collision_counts": [1, 0],
        "physics_anomaly_event_counts": [0, 0],
        "timeout_counts": [0, 1],
        "near_miss_counts": [2, 4],
        "actual_speed_means": [12.0, 14.0],
        "actual_speed_variances": [1.5, 2.5],
        "target_speed_means": [19.0, 19.0],
        "competition_summaries": [
            {
                "route_intent_summary": {
                    "intent_distribution": {"left": 1, "keep-current": 1},
                    "keep_current_rate": 0.5,
                    "mask_legal_fraction": 0.5,
                    "mean_mask_sparsity": 0.5,
                    "direction_preserved_violations": 0,
                    "fallback_count": 0,
                    "mean_candidate_count_total": 9.0,
                    "mean_candidate_count_same_exit_direction": 3.0,
                }
            },
            {"route_intent_summary": {"intent_distribution": {"keep-current": 2}}},
        ],
    }
    with (seed / "metrics_history_ep2.pkl").open("wb") as f:
        pickle.dump(history, f)

    summary = summarize_cell([seed])
    assert summary["success_rate"]["median"] == 62.5
    assert summary["collision_rate"]["median"] == 25.0
    assert summary["route_intent_decision_count"]["median"] == 2.0
    assert summary["route_intent_direction_preserved_violations"]["median"] == 0.0
    assert summary["route_intent_same_destination_violations"]["median"] == 0.0
    assert "pareto_hypervolume" in summary
