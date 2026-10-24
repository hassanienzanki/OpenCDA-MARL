"""Two "runs cost more than they should" defects.

* ``record`` traffic mode re-drew and re-saved the replay file on every episode
  reset. The coordinator also resets after the final episode, so the file on
  disk was a draw that had never been simulated (a 1-episode record run saved
  the 2nd draw). It now records once per run and replays that file afterwards.
* The matrix launchers always passed ``--apply_ml``, which loads YOLOv8 into
  every cell process (and downloads ~52 MB on a fresh checkout, racing between
  parallel slots) even though every matrix condition ships
  ``perception.activate: false`` and nothing in the MARL path reads the model.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts" / "experiments"


def _launcher():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import importlib
    return importlib.import_module("launch_matrix")


def _traffic_manager(tmp_path, monkeypatch, mode):
    from opencda_marl._kernel.traffic.traffic_manager import MARLTrafficManager

    tm = object.__new__(MARLTrafficManager)
    tm.mode = mode
    tm.replay_file = str(tmp_path / "traffic.json")
    tm._recording_saved = False
    tm._events = []
    tm._event_traces = []
    tm.preference_sampler = None
    calls = []
    tm._generate_and_save_events = lambda: calls.append("record") or tm.__dict__.__setitem__(
        "_recording_saved", True)
    tm._load_events = lambda: calls.append("load")
    tm._generate_events = lambda: calls.append("draw")
    return tm, calls


def test_record_mode_writes_the_file_once_then_replays_it(tmp_path, monkeypatch):
    from opencda_marl._kernel.traffic.traffic_manager import MARLTrafficManager

    tm, calls = _traffic_manager(tmp_path, monkeypatch, "record")
    MARLTrafficManager._initialize_events(tm)          # run start
    MARLTrafficManager.reset(tm)                        # after episode 1
    MARLTrafficManager.reset(tm)                        # after episode 2 (post-run reset)
    assert calls == ["record", "load", "load"], calls


def test_live_and_replay_modes_are_unchanged(tmp_path, monkeypatch):
    from opencda_marl._kernel.traffic.traffic_manager import MARLTrafficManager

    tm, calls = _traffic_manager(tmp_path, monkeypatch, "live")
    MARLTrafficManager._initialize_events(tm)
    MARLTrafficManager.reset(tm)
    assert calls == ["draw", "draw"], "live re-draws every episode"

    tm, calls = _traffic_manager(tmp_path, monkeypatch, "replay")
    MARLTrafficManager._initialize_events(tm)
    MARLTrafficManager.reset(tm)
    assert calls == ["load", "load"], "replay always loads"


def test_apply_ml_is_requested_only_when_perception_is_active(tmp_path):
    """_needs_ml merges the three layers the runtime merges, so a perception
    block that lives only in the family default is still seen."""
    lm = _launcher()
    # A real matrix condition: perception.activate comes from momarl/default.yaml.
    cell = REPO / "opencda_marl" / "_bundled" / "configs" / "momarl" / "intersection" / "speed_only_pcmappo.yaml"
    assert cell.exists()
    assert lm._needs_ml(cell) is False

    # A scenario whose family default activates perception must still ask for it.
    import yaml as _yaml
    default = _yaml.safe_load((REPO / "opencda_marl" / "_bundled" / "configs" / "momarl" / "default.yaml").read_text(encoding="utf-8"))
    found = []
    stack = [default]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if isinstance(node.get("perception"), dict):
                found.append(node["perception"])
            stack.extend(v for v in node.values() if isinstance(v, (dict, list)))
        elif isinstance(node, list):
            stack.extend(v for v in node if isinstance(v, (dict, list)))
    assert found, "the family default is where perception.activate lives"
    assert all(p.get("activate") is False for p in found), found


def _runtime_cell(tmp_path, monkeypatch, cell_text, activate=True, fragment=None):
    """A runtime cell under a temporary runtime root, with its own family
    default (perception on or off) and scenario base behind it."""
    lm = _launcher()
    monkeypatch.delenv("OPENCDA_MARL_CONFIG_ROOTS", raising=False)
    monkeypatch.setattr(lm, "RUNTIME_ROOT", tmp_path)
    family = tmp_path / "momarl"
    scenario = family / "ml_gate_scenario"
    scenario.mkdir(parents=True)
    (family / "default.yaml").write_text(yaml.safe_dump(
        {"agents": {"vehicle": {"sensing": {"perception": {"activate": activate}}}}}),
        encoding="utf-8")
    (scenario / "base.yaml").write_text("scenario: {}\n", encoding="utf-8")
    if fragment is not None:
        (scenario / "frag.yaml").write_text(yaml.safe_dump(fragment), encoding="utf-8")
    cell = scenario / "ml_gate_cell.yaml"
    cell.write_text(cell_text, encoding="utf-8")
    return lm, cell


def test_an_empty_section_in_the_cell_does_not_hide_the_default(tmp_path, monkeypatch):
    """The runtime keeps the lower layer's mapping under an empty key, so the
    scan must too: a plain merge read the cell's empty ``agents:`` section as
    erasing the default's ``perception.activate: true`` and dropped
    ``--apply_ml`` for a run that has perception on."""
    for cell_text in ("agents:\n  vehicle:\n", "agents:\n"):
        lm, cell = _runtime_cell(tmp_path / str(len(cell_text)), monkeypatch, cell_text)
        assert lm._needs_ml(cell) is True, cell_text

    lm, cell = _runtime_cell(tmp_path / "off", monkeypatch, "agents:\n  vehicle:\n",
                             activate=False)
    assert lm._needs_ml(cell) is False


def test_a_cell_include_that_activates_perception_needs_the_model(tmp_path, monkeypatch):
    """``include:`` fragments compose into the cell at runtime, so the scan
    sees a perception switch that only a fragment sets."""
    fragment = {"agents": {"vehicle": {"sensing": {"perception": {"activate": True}}}}}
    lm, cell = _runtime_cell(tmp_path, monkeypatch, "include:\n  - frag\n",
                             activate=False, fragment=fragment)
    assert lm._needs_ml(cell) is True


def test_unreadable_config_keeps_the_old_behaviour(tmp_path):
    lm = _launcher()
    assert lm._needs_ml(tmp_path / "does-not-exist.yaml") is True


def test_matrix_conditions_do_not_need_the_model():
    """Every shipped matrix condition resolves to perception off."""
    lm = _launcher()
    root = REPO / "opencda_marl" / "_bundled" / "configs" / "momarl" / "intersection"
    checked = 0
    for name in sorted(set(lm.CONDITION_CONFIGS.values())):
        cfg = root / name
        if not cfg.exists():
            continue
        checked += 1
        assert lm._needs_ml(cfg) is False, f"{name} would load YOLO"
    assert checked > 0, "no condition configs found"


# --- where a recording lands -------------------------------------------------
# A recording is produced by a run, so it belongs with the run's outputs.
# Writing it next to the packaged ground truth (or into whatever directory the
# process happened to start in) invited a draw from some experiment to be
# mistaken for the benchmark; promotion into opencda_marl/_bundled/recordings/
# is now a deliberate copy.

def _recorder(replay_file, record_root=None):
    from opencda_marl._kernel.traffic.traffic_manager import MARLTrafficManager

    tm = object.__new__(MARLTrafficManager)
    tm.replay_file = str(replay_file)
    if record_root is not None:
        tm.record_root = Path(record_root)
    return tm


def test_record_root_defaults_under_artifacts():
    from opencda_marl._kernel.traffic.traffic_manager import resolve_record_root

    assert resolve_record_root() == Path("artifacts") / "recordings"
    assert resolve_record_root({}) == Path("artifacts") / "recordings"


def test_record_root_follows_the_outputs_config():
    from opencda_marl._kernel.traffic.traffic_manager import resolve_record_root

    assert resolve_record_root({"root_dir": "runs/exp1"}) == Path("runs/exp1") / "recordings"
    assert resolve_record_root(
        {"root_dir": "runs/exp1", "recordings": {"dir": "traffic"}}) == Path("runs/exp1") / "traffic"


def test_a_relative_replay_file_records_under_the_record_root(tmp_path):
    """Only the basename of `replay_file` carries over — not its directory."""
    tm = _recorder("recordings/traffic_200vph.json", record_root=tmp_path / "artifacts" / "recordings")
    assert tm._record_destination() == (tmp_path / "artifacts" / "recordings" / "traffic_200vph.json").resolve()


def test_an_absolute_replay_file_is_honoured(tmp_path):
    target = (tmp_path / "elsewhere" / "mine.json").resolve()
    assert _recorder(target, record_root=tmp_path / "artifacts")._record_destination() == target


def test_a_bare_recorder_still_resolves_without_a_configured_root():
    tm = _recorder("recordings/traffic_200vph.json")          # class-level default
    assert tm._record_destination() == (Path("artifacts") / "recordings" / "traffic_200vph.json").resolve()


def test_the_saved_file_is_what_later_episodes_replay(tmp_path):
    """After recording, `replay_file` points at the file on disk.

    Otherwise a relative name would re-resolve on the next episode and could
    load the packaged recording of the same name instead of this run's draw.
    """
    from opencda_marl._kernel.traffic.traffic_manager import MARLTrafficManager

    tm = _recorder("recordings/traffic_200vph.json",
                   record_root=tmp_path / "artifacts" / "recordings")
    tm._events = ["one event"]
    tm._recording_saved = False
    tm.config = {}
    tm.flows = []
    tm.vbp = []
    tm._generate_events = lambda: None
    saved = {}

    class _Serializer:
        @staticmethod
        def export_events_to_json(events, path):
            saved["path"] = path
            Path(path).write_text("[]", encoding="utf-8")
            return True

    import opencda_marl._kernel.traffic.traffic_manager as tm_mod
    original = tm_mod.EventSerializer
    tm_mod.EventSerializer = _Serializer
    try:
        MARLTrafficManager._generate_and_save_events(tm)
    finally:
        tm_mod.EventSerializer = original

    expected = (tmp_path / "artifacts" / "recordings" / "traffic_200vph.json").resolve()
    assert Path(saved["path"]) == expected          # written under the record root
    assert Path(tm.replay_file) == expected         # and replayed from there
    assert tm._recording_saved is True
