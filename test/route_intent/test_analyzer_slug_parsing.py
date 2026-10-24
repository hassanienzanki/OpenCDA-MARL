"""How the matrix analyzers parse cell slugs.

Analyzers must not mistake an evaluation cell for the training cell it came
from. An eval slug is a training slug plus a suffix (`..._seed0_eval200vph`).
Parsed with ``re.match``, which matches a prefix, the eval directory produced
the same ``(action, algo, regime, seed)`` key as its training cell, so an eval
cell's outcome replaced its training cell's in every aggregate keyed by that
tuple. The training-side parsers therefore use ``fullmatch``.

The slug prefix is whatever ``--slug-prefix`` a run used: the analyzers parse
cells under any prefix (the launcher default, an explicit one, or the default
of an earlier version), so a non-default prefix never yields an empty table.
The prefix names the run, so it stays in every row and comparison group:
cells of two runs under one root are never merged, and ``--slug-prefix``
selects one run.
"""
import importlib
import json
import pickle
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


def _slug(prefix=None, *, condition="speed_only_pcmappo", regime="aligned", seed=0):
    common = _mod("matrix_common")
    return common.slug_for(
        map_slug="custom", condition=condition, regime=regime, seed=seed,
        prefix=common.DEFAULT_SLUG_PREFIX if prefix is None else prefix,
    )


TRAIN = _slug()
EVAL = TRAIN + "_eval200vph"

# Prefixes a run may carry: the default, none, an explicit two-part tag like
# an earlier default, and one with characters that are special in a regex.
PREFIXES = [None, "", "earlier_tag_", "run-2[a]+_"]


def test_training_analyzer_rejects_eval_slugs():
    m = _mod("analyze_matrix_results")
    parsed = m.parse_cell_name(TRAIN)
    assert parsed and parsed["action"] == "speed_only" and parsed["algo"] == "pcmappo"
    assert parsed["regime"] == "aligned" and parsed["seed"] == "0"
    assert m.parse_cell_name(EVAL) is None, "eval cell must not parse as a training cell"


def test_route_diversity_inspector_rejects_eval_slugs():
    m = _mod("inspect_route_diversity")
    assert m.SLUG_RE.fullmatch(TRAIN) is not None
    assert m.SLUG_RE.fullmatch(EVAL) is None


def test_eval_runner_rejects_already_evaluated_slugs():
    """run_matrix_eval walks the checkpoint tree, which also holds eval cells;
    re-evaluating one would produce `..._eval200vph_eval200vph`."""
    m = _mod("run_matrix_eval")
    assert m.parse_slug(TRAIN) is not None
    assert m.parse_slug(EVAL) is None


def test_eval_runner_parses_exactly_the_maps_with_a_profile():
    """The runner regenerates each cell's profile from its map, so a slug
    naming a map without a profile is skipped as unparseable instead of
    failing the whole pass when the profile lookup raises. The map
    alternation is the profile table itself: no extra map is hard-coded."""
    common = _mod("matrix_common")
    m = _mod("run_matrix_eval")
    alternatives = re.search(r"\(\?P<map>([^)]*)\)", m.SLUG_RE.pattern).group(1)
    assert sorted(alternatives.split("|")) == sorted(re.escape(k) for k in common.MAP_PROFILES)
    for map_slug in common.MAP_PROFILES:
        slug = common.slug_for(map_slug=map_slug, condition="learn_path_pcmappo",
                               regime="aligned", seed=0)
        assert m.parse_slug(slug)["map"] == map_slug
    unknown = common.slug_for(map_slug="nomap", condition="learn_path_pcmappo",
                              regime="aligned", seed=0)
    assert "nomap" not in common.MAP_PROFILES
    assert m.parse_slug(unknown) is None


def test_eval_analyzer_still_parses_eval_slugs():
    """The eval analyzer reads the eval tree: there the suffix is expected."""
    m = _mod("analyze_matrix_eval")
    parsed = m.parse_slug(EVAL)
    assert parsed is not None, "the eval analyzer must accept eval slugs"
    assert parsed["action"] == "speed_only" and parsed["seed"] == "0"


@pytest.mark.parametrize("slug", [
    _slug(condition="learn_path_pcmappo", regime="conflicting", seed=3),
    _slug(condition="rule_path_pcmappo", regime="aligned", seed=1),
])
def test_launcher_training_slugs_parse(slug):
    assert _mod("analyze_matrix_results").parse_cell_name(slug) is not None


@pytest.mark.parametrize("prefix", PREFIXES)
def test_every_parser_accepts_any_slug_prefix(prefix):
    """The launcher's slug for one cell parses in every analyzer whatever the
    prefix, and the parsed fields do not depend on it."""
    slug = _slug(prefix, condition="learn_path_pcmappo", regime="conflicting", seed=2)
    expected_prefix = _mod("matrix_common").DEFAULT_SLUG_PREFIX if prefix is None else prefix

    results = _mod("analyze_matrix_results").parse_cell_name(slug)
    inspector = _mod("inspect_route_diversity").SLUG_RE.fullmatch(slug)
    runner = _mod("run_matrix_eval").parse_slug(slug)
    evaluator = _mod("analyze_matrix_eval").parse_slug(slug + "_eval200vph")
    for parsed in (results, inspector and inspector.groupdict(), evaluator):
        assert parsed is not None
        assert parsed["prefix"] == expected_prefix
        assert (parsed["action"], parsed["algo"]) == ("learn_path", "pcmappo")
        assert (parsed["regime"], parsed["seed"]) == ("conflicting", "2")
    assert runner is not None and runner["prefix"] == expected_prefix
    assert runner["condition"] == "learn_path_pcmappo" and runner["map"] == "custom"


def test_short_names_keep_the_prefix_drop_the_map_and_keep_the_seed():
    slug = _slug("earlier_tag_", seed=3)
    assert _mod("analyze_matrix_results").short_cell_name(slug) == \
        "earlier_tag_speed_only_pcmappo_aligned_seed3"
    assert _mod("analyze_matrix_eval").short_slug(slug + "_eval200vph") == \
        "earlier_tag_speed_only_pcmappo_aligned_seed3_eval200vph"
    assert _mod("inspect_route_diversity").short_name(slug) == \
        "earlier_tag_speed_only_pcmappo_aligned_seed3"
    # An empty prefix leaves just the condition; a non-cell prints unchanged.
    assert _mod("analyze_matrix_results").short_cell_name(_slug("", seed=3)) == \
        "speed_only_pcmappo_aligned_seed3"
    assert _mod("analyze_matrix_results").short_cell_name("notes") == "notes"


def _write_training_cell(root, name, success, collision):
    record = {"success_counts": [success], "collision_counts": [collision],
              "timeout_counts": [0], "rewards": [1.0], "utilities": [2.0]}
    cell_dir = root / name
    cell_dir.mkdir()
    with (cell_dir / "metrics_history_ep1.pkl").open("wb") as fh:
        pickle.dump(record, fh)


def test_training_analyzer_loads_cells_under_any_prefix(tmp_path, monkeypatch):
    m = _mod("analyze_matrix_results")
    cells = [_slug(None), _slug("earlier_tag_", seed=1)]
    for name in cells + [_slug(None) + "_eval200vph"]:
        _write_training_cell(tmp_path, name, 3, 1)
    (tmp_path / "stray.txt").write_text("not a cell", encoding="utf-8")

    monkeypatch.setattr(m, "MATRIX_METRICS", tmp_path)
    loaded = m.load_cells()
    assert set(loaded) == set(cells)
    assert loaded[cells[1]]["stats"]["succ_mean"] == pytest.approx(75.0)

    monkeypatch.setattr(m, "MATRIX_METRICS", tmp_path / "missing")
    assert m.load_cells() == {}


# One cell (speed_only, pcmappo, aligned, seed 0) from two runs: a run under
# an earlier prefix with 100 % success and the current run with 25 %.
TWO_RUNS = {"earlier_tag_": (4, 0), "matrix_": (1, 3)}


def _two_run_training_root(tmp_path):
    for prefix, (success, collision) in TWO_RUNS.items():
        _write_training_cell(tmp_path, _slug(prefix), success, collision)
    _write_training_cell(
        tmp_path, _slug("matrix_", condition="learn_path_pcmappo"), 2, 2)
    return tmp_path


def test_two_runs_of_one_cell_stay_apart_in_the_training_analyzer(tmp_path, monkeypatch, capsys):
    """Cells of two runs print as two distinct rows and form two Tier 3
    groups, so the learn_path delta is taken against its own run's
    speed_only, never against the other run's."""
    m = _mod("analyze_matrix_results")
    monkeypatch.setattr(m, "MATRIX_METRICS", _two_run_training_root(tmp_path))
    cells = m.load_cells()

    shorts = {m.short_cell_name(_slug(prefix)) for prefix in TWO_RUNS}
    assert len(shorts) == 2

    groups = m.tier3_groups(cells)
    assert set(groups) == {("earlier_tag_", "pcmappo", "aligned"),
                           ("matrix_", "pcmappo", "aligned")}
    assert groups[("matrix_", "pcmappo", "aligned")]["speed_only"]["succ_mean"] == \
        pytest.approx(25.0)
    assert groups[("earlier_tag_", "pcmappo", "aligned")]["speed_only"]["succ_mean"] == \
        pytest.approx(100.0)
    assert "learn_path" not in groups[("earlier_tag_", "pcmappo", "aligned")]

    m.tier3_action_space_comparison(cells)
    out = capsys.readouterr().out
    # learn_path 50 % against its own run's speed_only 25 %, not 100 %.
    assert "delta learn_path vs speed_only: succ +25.0pp" in out


def test_training_analyzer_filters_one_run_and_warns_on_mixed(tmp_path, monkeypatch, capsys):
    m = _mod("analyze_matrix_results")
    monkeypatch.setattr(m, "MATRIX_METRICS", _two_run_training_root(tmp_path))
    monkeypatch.setattr(m, "MATRIX_CKPTS", tmp_path / "no_checkpoints")
    assert set(m.load_cells("earlier_tag_")) == {_slug("earlier_tag_")}
    assert len(m.load_cells("matrix_")) == 2
    assert m.load_cells("absent_") == {}

    m.main([])
    out = capsys.readouterr().out
    assert "WARN: cells from 2 slug prefixes ('earlier_tag_', 'matrix_')" in out
    m.main(["--slug-prefix", "matrix_"])
    assert "WARN" not in capsys.readouterr().out


def _write_eval_cell(root, name, success):
    cell_dir = root / name
    cell_dir.mkdir()
    (cell_dir / "eval_summary.json").write_text(json.dumps({
        "status": "completed", "completed_episodes": 1,
        "overall": {"mean_success_rate": success, "mean_collision_rate": 0.0},
    }), encoding="utf-8")


def test_two_runs_of_one_cell_stay_apart_in_the_eval_analyzer(tmp_path, capsys):
    m = _mod("analyze_matrix_eval")
    for prefix, success in (("earlier_tag_", 100.0), ("matrix_", 25.0)):
        _write_eval_cell(tmp_path, _slug(prefix) + "_eval200vph", success)
    (tmp_path / "notes").mkdir()

    cells = m.load_cells(tmp_path)
    assert len(cells) == 3          # without a filter every directory loads
    shorts = {m.short_slug(_slug(p) + "_eval200vph") for p in ("earlier_tag_", "matrix_")}
    assert len(shorts) == 2
    groups = m.tier3_groups(cells)
    assert set(groups) == {("earlier_tag_", "pcmappo", "aligned"),
                           ("matrix_", "pcmappo", "aligned")}

    only = m.load_cells(tmp_path, "matrix_")
    assert set(only) == {_slug("matrix_") + "_eval200vph"}

    m.main(["--eval-dir", str(tmp_path)])
    assert "WARN: cells from 2 slug prefixes" in capsys.readouterr().out
    m.main(["--eval-dir", str(tmp_path), "--slug-prefix", "matrix_"])
    assert "WARN" not in capsys.readouterr().out


@pytest.mark.parametrize("prefix", PREFIXES)
def test_a_run_prefix_keeps_its_continuation_cells_in_the_eval_analyzer(tmp_path, prefix):
    """--slug-prefix reads the same cells of a run as a load without it,
    continuation cells included; the tier tables still skip those."""
    m = _mod("analyze_matrix_eval")
    run = _mod("matrix_common").DEFAULT_SLUG_PREFIX if prefix is None else prefix
    fresh = _slug(prefix, condition="learn_path_pcmappo") + "_eval200vph"
    continued = _slug(prefix, condition="continue_learn_path_pcmappo") + "_eval200vph"
    other = _slug("other_run_", condition="continue_learn_path_pcmappo") + "_eval200vph"
    for name in (fresh, continued, other):
        _write_eval_cell(tmp_path, name, 50.0)

    assert set(m.load_cells(tmp_path, run)) == {fresh, continued}
    assert m.slug_prefix_of(continued) == run
    assert m.parse_slug(continued) is None
    assert set(m.tier3_groups(m.load_cells(tmp_path, run))) == {(run, "pcmappo", "aligned")}


@pytest.mark.parametrize("prefix", PREFIXES + ["x_"])
def test_continuation_slugs_parse_as_their_own_condition(prefix):
    """A continuation slug (``<prefix>custom_continue_<action>_<algo>_...``)
    parses in the eval runner as the continuation condition and in the eval
    analyzer as a cell of its run; no parser reads it as the fresh cell it
    starts from, and the lazy prefix never absorbs the ``continue_`` part."""
    run = _mod("matrix_common").DEFAULT_SLUG_PREFIX if prefix is None else prefix
    slug = _slug(prefix, condition="continue_learn_path_pcmappo")

    runner = _mod("run_matrix_eval").parse_slug(slug)
    assert runner is not None
    assert (runner["prefix"], runner["map"]) == (run, "custom")
    assert runner["condition"] == "continue_learn_path_pcmappo"

    evaluator = _mod("analyze_matrix_eval")
    assert evaluator.slug_prefix_of(slug + "_eval200vph") == run
    assert evaluator.parse_slug(slug + "_eval200vph") is None
    assert _mod("analyze_matrix_results").parse_cell_name(slug) is None
    assert _mod("inspect_route_diversity").SLUG_RE.fullmatch(slug) is None


def test_route_inspector_and_eval_runner_select_one_run(tmp_path):
    names = [_slug(p, condition="learn_path_pcmappo") for p in ("earlier_tag_", "matrix_", "")]
    for name in names + [_slug("matrix_")]:    # speed_only: no route head
        (tmp_path / name).mkdir()

    inspector = _mod("inspect_route_diversity")
    assert [n for n, _ in inspector.route_head_cells(tmp_path)] == sorted(names)
    assert [n for n, _ in inspector.route_head_cells(tmp_path, "matrix_")] == [names[1]]
    assert [n for n, _ in inspector.route_head_cells(tmp_path, "")] == [names[2]]

    runner = _mod("run_matrix_eval")
    every = sorted(names + [_slug("matrix_"), "notes"])
    assert runner.select_slugs(every) == every
    assert runner.select_slugs(every, slug_prefix="earlier_tag_") == [names[0]]
    assert runner.select_slugs(every, cells=[names[1], "notes"], slug_prefix="matrix_") == \
        [names[1]]


def test_mixed_prefix_warning():
    warn = _mod("matrix_common").mixed_prefix_warning
    assert warn([]) is None
    assert warn(["matrix_", "matrix_"]) is None
    assert warn(["matrix_", "", "matrix_"]) == (
        "WARN: cells from 2 slug prefixes ('', 'matrix_'); "
        "pass --slug-prefix to analyze one run")
