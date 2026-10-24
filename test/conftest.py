"""Shared pytest setup.

Makes the repository root importable no matter how pytest is invoked (some
test files historically did their own ``sys.path`` insertion, which made
imports order-dependent), then imports ``torch`` before ``numpy`` to avoid
DLL load-order errors on Windows pixi envs. Importing torch here at
collection time fixes the order for the whole pytest process regardless of
test file ordering.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: F401,E402  (import-order side effect)

# Headless plotting for the whole session. The vendored localization debug
# helper calls matplotlib.use('TkAgg') when show_animation is on (a global
# switch), and the pixi Windows env has no usable Tk, so any later test that
# creates a figure died with a TclError. Tests must never open a GUI anyway.
import os  # noqa: E402

os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib  # noqa: E402

matplotlib.use('Agg', force=True)

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _tensorboard_off_unless_configured(monkeypatch):
    """Learners default to TensorBoard ON with ``log_dir='runs'`` (relative to
    the CWD), so every test that builds an algorithm without a ``tensorboard``
    block used to leave an event file under ``<repo>/runs``. Tests that want
    the writer say so explicitly (``tensorboard: {enabled: True, log_dir: ...}``);
    everything else runs without one.
    """
    from opencda_marl.builtin.algorithms import training_logger as tl

    original = tl.TrainingLogger._init_tensorboard

    def _init_without_default_writer(self, config):
        if isinstance(config, dict) and config.get('tensorboard') is None:
            config = dict(config)
            config['tensorboard'] = {'enabled': False}
        return original(self, config)

    monkeypatch.setattr(tl.TrainingLogger, '_init_tensorboard', _init_without_default_writer)

    # Same rule for the metrics store (default ON, relative 'metrics' dir).
    original_store = tl.TrainingLogger._init_metrics_store

    def _init_without_default_store(self, config):
        # Opt-in only: MARLManager merges a default-ON block into every
        # learner config, so "absent" is not a usable signal here.
        block = config.get('metrics_store') if isinstance(config, dict) else None
        explicit = isinstance(block, dict) and block.get('enabled') is True
        if isinstance(config, dict) and not explicit:
            config = dict(config)
            config['metrics_store'] = {'enabled': False}
        return original_store(self, config)

    monkeypatch.setattr(tl.TrainingLogger, '_init_metrics_store', _init_without_default_store)

    # ...and for the manager-owned store of external policies.
    from opencda_marl._kernel.policy import manager as mm

    original_external = mm.MARLManager._init_external_metrics_store

    def _init_external_only_when_asked(self):
        outputs = self.config.get('outputs') if hasattr(self.config, 'get') else None
        block = (outputs or {}).get('metrics_store') if isinstance(outputs, dict) else None
        if isinstance(block, dict) and block.get('enabled') is True:
            return original_external(self)
        self._external_store = None
        return None

    monkeypatch.setattr(mm.MARLManager, '_init_external_metrics_store', _init_external_only_when_asked)


@pytest.fixture(autouse=True)
def _plugin_registry_never_frozen():
    """A coordinator built inside one test freezes the plugin registry;
    the next test may legitimately register fixtures. The freeze is a
    run-stability guard, not a test subject — reset it around every test
    (the dedicated freeze test exercises the mechanism explicitly)."""
    from opencda_marl.plugins import registry as plugin_registry

    plugin_registry._frozen = False
    yield
    plugin_registry._frozen = False
