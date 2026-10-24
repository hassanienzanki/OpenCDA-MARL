"""Config, both directions: profile includes and per-component validation."""
import pytest

from opencda_marl.utils.config_locator import load_profile_with_includes


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_profile_composes_fragments_in_order(tmp_path):
    _write(tmp_path / "fragments" / "dense.yaml",
           "scenario:\n  simulation:\n    max_steps: 2400\nworld:\n  seed: 1\n")
    _write(tmp_path / "fragments" / "short.yaml",
           "scenario:\n  simulation:\n    max_steps: 600\n")
    _write(tmp_path / "profile.yaml",
           "include:\n  - fragments/dense\n  - fragments/short\n"
           "world:\n  seed: 42\n")

    cfg = load_profile_with_includes(tmp_path / "profile.yaml")
    # later fragment overrides earlier; the profile overrides them all
    assert cfg.scenario.simulation.max_steps == 600
    assert cfg.world.seed == 42
    assert "include" not in cfg


def test_fragments_may_include_fragments_and_cycles_fail(tmp_path):
    _write(tmp_path / "a.yaml", "include: [b]\nx: 1\n")
    _write(tmp_path / "b.yaml", "y: 2\n")
    cfg = load_profile_with_includes(tmp_path / "a.yaml")
    assert cfg.x == 1 and cfg.y == 2

    _write(tmp_path / "c.yaml", "include: [d]\n")
    _write(tmp_path / "d.yaml", "include: [c]\n")
    with pytest.raises(ValueError, match="include cycle"):
        load_profile_with_includes(tmp_path / "c.yaml")


def test_missing_include_fails_loudly(tmp_path):
    _write(tmp_path / "p.yaml", "include: [nope]\n")
    with pytest.raises(FileNotFoundError, match="nope"):
        load_profile_with_includes(tmp_path / "p.yaml")


def test_component_validator_runs_at_resolution():
    from opencda_marl.api import components as comp

    before = ({k: dict(v) for k, v in comp._COMPONENTS.items()},
              {k: dict(v) for k, v in comp._COMPONENT_SOURCES.items()},
              {k: dict(v) for k, v in comp._COMPONENT_VALIDATORS.items()})
    try:
        def _validate(cfg):
            if cfg.get("threshold", 0) <= 0:
                raise ValueError("threshold must be positive")

        comp.register_component("reward_calculator", "picky",
                                validate=_validate)(object)
        # bad config: the plugin's own message surfaces at resolve time
        with pytest.raises(ValueError, match="threshold must be positive"):
            comp.resolve_component("reward_calculator", "picky",
                                   config={"threshold": 0})
        # good config resolves; no config skips validation (late binding)
        assert comp.resolve_component("reward_calculator", "picky",
                                      config={"threshold": 5}) is object
        assert comp.resolve_component("reward_calculator", "picky") is object
    finally:
        comp._COMPONENTS.clear(); comp._COMPONENTS.update({k: dict(v) for k, v in before[0].items()})
        comp._COMPONENT_SOURCES.clear(); comp._COMPONENT_SOURCES.update({k: dict(v) for k, v in before[1].items()})
        comp._COMPONENT_VALIDATORS.clear(); comp._COMPONENT_VALIDATORS.update({k: dict(v) for k, v in before[2].items()})
