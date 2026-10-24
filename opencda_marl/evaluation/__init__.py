"""Unified measurement pipeline for cooperative-driving experiments.

Exports resolve lazily so a slim SUMO-only install (the engine wheel)
pays only for what it imports. Metrics are looked at in the web monitor;
the manager mirrors evaluation summaries into the metrics store.
"""

__all__ = [
    "EvaluationManager",
    "TrainingMetrics",
    "summarize_intent_decisions",
]

_LAZY = {
    "EvaluationManager": ("opencda_marl.evaluation.manager", "EvaluationManager"),
    "TrainingMetrics": ("opencda_marl.evaluation.training_metrics", "TrainingMetrics"),
    "summarize_intent_decisions": ("opencda_marl.evaluation.objective_metrics",
                                   "summarize_intent_decisions"),
}


def __getattr__(name):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(target[0]), target[1])
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
