"""Training telemetry: a small SQLite metrics store and the monitor that reads it.

TensorBoard's event files are append-only protobuf logs: reading the tail of
a long run means re-parsing the whole file, the UI stalls once a run reaches
tens of thousands of scalars, and the data cannot be queried. The
:class:`MetricsStore` keeps one SQLite file per run (WAL mode, so a live
writer never blocks a reader), buffers writes into short transactions, and
answers step-range / downsampled queries directly. Every learner writes to it
through the learner package's ``TrainingLogger``; the
monitor (``python -m opencda_marl.telemetry.monitor``) serves the runs under
an artifacts root to a browser with no external dependencies.
"""
from .store import MetricsStore, RunInfo, discover_runs

__all__ = ["MetricsStore", "RunInfo", "discover_runs"]
