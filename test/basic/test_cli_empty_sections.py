"""The wheel CLI reads a section left empty as absent before the coordinator.

``opencda-marl run`` checks ``runtime.backend`` on the merged config before
it builds the coordinator, which is where the run's sections are otherwise
normalised. A ``runtime:`` left empty, with no layer below holding a
mapping, reached that check as None and raised ``AttributeError`` instead of
the backend message.
"""
import argparse

from opencda_marl.utils.config_locator import MARL_CONFIG_ROOTS_ENV


def _family(tmp_path, profile_text):
    family = tmp_path / "configs" / "sumo-marl"
    (family / "intersection").mkdir(parents=True)
    (family / "default.yaml").write_text("MARL:\n  training: {}\n", encoding="utf-8")
    (family / "intersection" / "base.yaml").write_text(
        "scenario:\n  type: intersection\n", encoding="utf-8")
    (family / "intersection" / "p.yaml").write_text(profile_text, encoding="utf-8")


def test_an_empty_runtime_section_reads_as_absent(tmp_path, monkeypatch, capsys):
    from opencda_marl.cli import _build_config, _cmd_run

    monkeypatch.delenv(MARL_CONFIG_ROOTS_ENV, raising=False)
    _family(tmp_path, "runtime:\noutputs:\n")
    monkeypatch.chdir(tmp_path)

    cfg = _build_config("sumo-marl/intersection/p", [])
    assert cfg.runtime == {} and cfg.outputs == {}

    args = argparse.Namespace(test_scenario="sumo-marl/intersection/p",
                              config_root=[], debug=False)
    # No backend is configured, so the run is a CARLA one, which the bare
    # wheel refuses with its own message rather than a traceback.
    assert _cmd_run(args) == 2
    assert "'carla' backend" in capsys.readouterr().err
