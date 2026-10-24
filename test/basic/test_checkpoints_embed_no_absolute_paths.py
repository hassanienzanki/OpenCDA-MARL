"""Bundled checkpoints carry no absolute filesystem paths.

Every learner pickles its configuration into its checkpoint, and that
configuration can hold output directories. A path rooted in the machine
that trained the weights would leak into every copy of the package, so each
bundled ``.pth`` is opened as the zip archive ``torch.save`` writes and
every member is checked. The pickles are rebuilt by an unpickler that turns
every class they name into an inert record, so nothing in the archive is
imported or run, and every string reachable from the result is checked; a
``pathlib`` object is joined back from its parts first, so a path stored as
a ``Path`` is checked whole. The small text members are checked as text.
Tensor storage members are raw numbers and are skipped; a path pattern
there would be noise, not a path.

The same check runs on what every built-in learner saves today: each one
is built with its run outputs under an absolute directory, the way a run
configures it, and its checkpoint must hold none of them (tabular
Q-learning pickles its checkpoint directly, so that file is one pickle).
A run hands a learner its section of the whole run config, merged by the
manager: a plain dict whose list and dict values are still OmegaConf
nodes, and each node keeps a reference to the whole run config. Pickled
as is, any one of them carries every section into the checkpoint, so the
learners are also built that way and their checkpoints must name no
OmegaConf class at all.
"""
from __future__ import annotations

import importlib
import io
import pickle
import re
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import SimpleNamespace

import pytest

from opencda_marl._bundled import BUNDLED_CHECKPOINTS

#: Any rooted path: a drive path (``X:\...`` or ``X:/...``), a POSIX path of
#: two or more segments under any root, a UNC path in either slash style
#: (``//host/share``, ``\\host\share``), or any of them as a ``file:`` URL.
#: The look-behinds keep relative paths such as ``checkpoints/data/x``,
#: ``./var/run`` or ``km/h`` and network URLs such as ``http://host/x`` out.
_ABSOLUTE_PATH = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]+\w"
    r"|(?<![\w.~:/-])/+[\w.-]+/[\w.-]"
    r"|\\\\[\w.-]+\\"
    r"|(?<![\w.+-])file:/+[\w.-]+/[\w.-]"
)

_STORAGE_MEMBER = re.compile(r"(^|/)data/\d+$")

_CHECKPOINTS = sorted(BUNDLED_CHECKPOINTS.rglob("*.pth"))


class _Inert:
    """Stands in for every class a pickle names and records what the
    pickle hands it, so the strings inside stay reachable."""

    pickled_as = ("", "")

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        obj.args, obj.kwargs, obj.state, obj.items = args, kwargs, None, []
        return obj

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.state = state

    def __setitem__(self, key, value):
        self.items.append((key, value))

    def append(self, value):
        self.items.append(value)

    def extend(self, values):
        self.items.extend(values)

    def add(self, value):
        self.items.append(value)

    def joined_path(self) -> str | None:
        """The whole path, with ``/`` separators, when this stands in for a
        ``pathlib`` object."""
        module, name = self.pickled_as
        if module.split(".")[0] != "pathlib" or self.kwargs:
            return None
        if not self.args or not all(isinstance(a, str) for a in self.args):
            return None
        flavour = PureWindowsPath if "Windows" in name else PurePosixPath
        return flavour(*self.args).as_posix()


class _InertUnpickler(pickle.Unpickler):
    """Resolves every class to an :class:`_Inert` record and every tensor
    storage reference to ``None``; nothing is imported."""

    def __init__(self, raw: bytes):
        super().__init__(io.BytesIO(raw))
        self._classes: dict[tuple[str, str], type] = {}

    def find_class(self, module, name):
        key = (module, name)
        if key not in self._classes:
            self._classes[key] = type("Inert", (_Inert,), {"pickled_as": key})
        return self._classes[key]

    def persistent_load(self, pid):
        return None


def _reachable_strings(root):
    seen, stack = set(), [root]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            yield item
            continue
        if isinstance(item, (bytes, bytearray)):
            yield bytes(item).decode("latin-1")
            continue
        if id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, dict):
            stack.extend(item.keys())
            stack.extend(item.values())
        elif isinstance(item, (list, tuple, set, frozenset)):
            stack.extend(item)
        elif isinstance(item, _Inert):
            joined = item.joined_path()
            if joined is not None:
                yield joined
            stack.extend((item.args, item.kwargs, item.state, item.items))


def _matches(member: str, texts) -> list[str]:
    return [f"{member}: {m.group(0)}"
            for text in texts for m in _ABSOLUTE_PATH.finditer(text)]


def _absolute_paths(path: Path) -> list[str]:
    """``member: match`` for every absolute path found in the archive, or
    in the file read as one pickle when it is not an archive."""
    if not zipfile.is_zipfile(path):
        return _matches(path.name, _reachable_strings(
            _InertUnpickler(path.read_bytes()).load()))
    found = []
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            if _STORAGE_MEMBER.search(member):
                continue
            raw = archive.read(member)
            texts = (_reachable_strings(_InertUnpickler(raw).load())
                     if member.endswith(".pkl")
                     else [raw.decode("latin-1")])
            found.extend(_matches(member, texts))
    return found


def _pickled_modules(path: Path) -> set[str]:
    """The module of every class the file's pickles name; nothing is
    imported."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            raws = [archive.read(m) for m in archive.namelist() if m.endswith(".pkl")]
    else:
        raws = [path.read_bytes()]
    modules = set()
    for raw in raws:
        unpickler = _InertUnpickler(raw)
        unpickler.load()
        modules.update(module for module, _ in unpickler._classes)
    return modules


def test_there_are_bundled_checkpoints_to_check():
    assert _CHECKPOINTS, f"no .pth found under {BUNDLED_CHECKPOINTS.name}/"


@pytest.mark.parametrize(
    "checkpoint", _CHECKPOINTS,
    ids=[p.relative_to(BUNDLED_CHECKPOINTS).parent.as_posix() for p in _CHECKPOINTS],
)
def test_bundled_checkpoint_embeds_no_absolute_path(checkpoint):
    assert zipfile.is_zipfile(checkpoint), "expected the zip format torch.save writes"
    assert _absolute_paths(checkpoint) == []


#: Roots for the flag samples; the pattern itself names none.
_SAMPLE_ROOTS = ("home", "Users", "data", "mnt", "var", "tmp", "opt", "srv", "root",
                 "any_root")


@pytest.mark.parametrize("text", [
    *(f"Z:{sep}example{sep}run" for sep in ("\\", "/")),
    *(f"/{root}/example/run" for root in _SAMPLE_ROOTS),
    "//host/share/run",
    "///example/run",
    "\\\\host\\share\\run",
    "file:///example/run",
    "file://host/share/run",
])
def test_the_pattern_flags_absolute_paths(text):
    assert _ABSOLUTE_PATH.search(text)


@pytest.mark.parametrize("text", [
    "checkpoints/sac/",
    "recordings/traffic.json",
    "metrics_history/data/run",
    "./var/run",
    "km/h",
    "Loss/actor",
    "http://127.0.0.1:8000/data/x",
    "https://example.org/a/b",
])
def test_the_pattern_ignores_relative_paths_and_urls(text):
    assert not _ABSOLUTE_PATH.search(text)


_POSIX_SAMPLE = "/home/example/runs"
_DRIVE_SAMPLE = "Z:/example/runs"


@pytest.mark.parametrize("log_dir, expected", [
    pytest.param(_POSIX_SAMPLE, 1, id="str"),
    pytest.param(PurePosixPath(_POSIX_SAMPLE), 1, id="posix-path"),
    pytest.param(PureWindowsPath(_DRIVE_SAMPLE), 1, id="windows-path"),
    # The second path reuses the first one's parts through the pickle memo.
    pytest.param([PurePosixPath(_POSIX_SAMPLE), PurePosixPath(_POSIX_SAMPLE + "2")],
                 2, id="memoised-parts"),
])
def test_a_path_in_a_saved_config_is_found(tmp_path, log_dir, expected):
    """End to end: a config holding an absolute path, saved the way the
    learners save theirs, is caught inside the archive's pickle."""
    torch = pytest.importorskip("torch")
    target = tmp_path / "checkpoint.pth"
    torch.save({"weights": torch.zeros(3),
                "config": {"tensorboard": {"log_dir": log_dir}}}, target)
    hits = _absolute_paths(target)
    assert len(hits) == expected, hits
    assert all("data.pkl: " in hit for hit in hits), hits


@pytest.mark.parametrize("sample", [_POSIX_SAMPLE, _DRIVE_SAMPLE],
                         ids=["posix", "drive"])
def test_a_path_in_a_saved_omegaconf_config_is_found(tmp_path, sample):
    """The learners' configs are OmegaConf containers, which hold a
    ``Path`` value (of the running platform's flavour) as a node of its
    own."""
    torch = pytest.importorskip("torch")
    omegaconf = pytest.importorskip("omegaconf")
    config = omegaconf.OmegaConf.create(
        {"tensorboard": {"log_dir": Path(sample)}})
    target = tmp_path / "checkpoint.pth"
    torch.save({"weights": torch.zeros(3), "config": config}, target)
    assert len(_absolute_paths(target)) == 1


def test_a_relative_path_object_is_not_flagged(tmp_path):
    torch = pytest.importorskip("torch")
    target = tmp_path / "checkpoint.pth"
    torch.save({"config": {"log_dir": PurePosixPath("checkpoints", "data", "run"),
                           "store": PureWindowsPath("metrics", "var")}}, target)
    assert _absolute_paths(target) == []


# --- what the built-in learners save --------------------------------------------

_SMALL_NETS = {"actor_hidden_dims": [16, 16], "critic_hidden_dims": [16, 16],
               "lstm_hidden_size": 8}

#: Every built-in learner that writes its config into its checkpoint, small
#: enough to build in milliseconds: module, class, hyperparameters,
#: state_dim, action_dim, and the file name its checkpoints take.
_LEARNERS = {
    "dqn": ("dqn", "DQNAlgorithm", {"speed_actions": [15.0, 30.0, 45.0], "hidden_dims": [16]},
            8, 3, "checkpoint.pth"),
    "q_learning": ("q_learning", "QLearningAlgorithm", {"speed_actions": [10, 20, 30]},
                   4, 3, "checkpoint.pkl"),
    "mappo": ("mappo", "MAPPOAlgorithm", dict(_SMALL_NETS), 5, 1, "checkpoint.pth"),
    "pcmappo": ("pcmappo", "PCMAPPOAlgorithm", {**_SMALL_NETS, "pref_dim": 2},
                5, 1, "checkpoint.pth"),
    "sac": ("sac", "SACAlgorithm", dict(_SMALL_NETS), 5, 1, "checkpoint.pth"),
    "td3": ("td3", "TD3Algorithm", dict(_SMALL_NETS), 5, 1, "checkpoint.pth"),
}

_RUN_OUTPUT_BLOCKS = {"tensorboard", "metrics_store"}


def _run_config(hyperparameters: dict, root: Path) -> dict:
    """``hyperparameters`` with the run-output blocks MARLManager merges in
    from ``outputs.*``, rooted at the absolute ``root`` (both disabled, so
    nothing is written there), and an absolute path outside those blocks."""
    return {
        **hyperparameters,
        "training_mode": True,
        "tensorboard": {"enabled": False, "log_dir": str(root / "runs"),
                        "run_name": "cell"},
        "metrics_store": {"enabled": False, "log_dir": str(root / "metrics"),
                          "run_name": "cell", "scenario": "family/scenario/profile"},
        "provenance": {"recording": root / "recordings" / "traffic.json"},
    }


def _saved_config(path: Path):
    if zipfile.is_zipfile(path):
        torch = pytest.importorskip("torch")
        return torch.load(path, map_location="cpu", weights_only=False)["config"]
    with open(path, "rb") as handle:
        return pickle.load(handle)["config"]


@pytest.mark.parametrize("name", sorted(_LEARNERS))
def test_a_learner_saves_its_config_without_absolute_paths(tmp_path, name):
    """A learner configured the way a run configures it saves a checkpoint
    with no absolute path anywhere and no run-output block, and with every
    hyperparameter as configured."""
    pytest.importorskip("torch")
    module, cls, hyperparameters, state_dim, action_dim, file_name = _LEARNERS[name]
    learner_cls = getattr(
        importlib.import_module(f"opencda_marl.builtin.algorithms.{module}"), cls)
    config = _run_config(hyperparameters, tmp_path)
    assert _ABSOLUTE_PATH.search(str(tmp_path)), "the sample root must be absolute"

    learner = learner_cls(config, state_dim, action_dim)
    target = tmp_path / file_name
    learner.save(str(target))

    assert target.is_file()
    assert _absolute_paths(target) == []
    saved = _saved_config(target)
    assert not _RUN_OUTPUT_BLOCKS & set(saved), sorted(saved)
    assert {k: saved[k] for k in hyperparameters} == hyperparameters


def _merged_by_the_manager(name: str, hyperparameters: dict, root: Path) -> dict:
    """The learner's config as a run builds it: its section of a whole run
    config, merged by MARLManager. The run outputs sit under the absolute
    ``root`` (both disabled, so nothing is written there), and an absolute
    path sits in a section the learner never reads."""
    omegaconf = pytest.importorskip("omegaconf")
    from opencda_marl._kernel.policy.manager import MARLManager

    run = omegaconf.OmegaConf.create({
        "world": {"weather_dir": str(root / "weather")},
        "outputs": {"root_dir": str(root), "tensorboard": {"enabled": False},
                    "metrics_store": {"enabled": False}},
        "MARL": {"algorithm": name, name: {
            **hyperparameters,
            "features": {"speed": 1, "rel_x": 1},
            # Nested, so it reaches the learner unresolved.
            "notes": {"tag": "run@${outputs.root_dir}"},
        }},
    })
    # The way the pipeline builder hands the manager its MARL block.
    marl = run.MARL.copy()
    marl["outputs"] = run.outputs
    return MARLManager._merge_shared_config(
        SimpleNamespace(config=marl, algorithm_name=name), marl[name])


@pytest.mark.parametrize("name", sorted(_LEARNERS))
def test_a_learner_configured_by_a_run_saves_none_of_the_run_config(tmp_path, name):
    """Every list and dict in the config a run builds is an OmegaConf node
    that references the whole run config. The checkpoint holds no such
    node, nothing else from that config, and no absolute path."""
    pytest.importorskip("torch")
    module, cls, hyperparameters, state_dim, action_dim, file_name = _LEARNERS[name]
    learner_cls = getattr(
        importlib.import_module(f"opencda_marl.builtin.algorithms.{module}"), cls)
    config = _merged_by_the_manager(name, hyperparameters, tmp_path)
    nodes = [k for k, v in config.items() if type(v).__module__.startswith("omegaconf.")]
    assert nodes, "the sample must hold OmegaConf nodes, as a run's config does"

    learner = learner_cls(config, state_dim, action_dim)
    target = tmp_path / file_name
    learner.save(str(target))

    assert target.is_file()
    assert _absolute_paths(target) == []
    assert not {m for m in _pickled_modules(target) if m.split(".")[0] == "omegaconf"}
    saved = _saved_config(target)
    assert not _RUN_OUTPUT_BLOCKS & set(saved), sorted(saved)
    assert {k: saved[k] for k in hyperparameters} == hyperparameters
    assert saved["notes"] == {"tag": "run@${outputs.root_dir}"}
