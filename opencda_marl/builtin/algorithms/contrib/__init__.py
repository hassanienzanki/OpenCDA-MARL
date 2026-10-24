"""Drop-in algorithms.

Put a module (or subpackage) here, register your learner in it with
:func:`opencda_marl.api.algorithms.register_algorithm`, and it is discovered
automatically the next time the engine loads this package — no pyproject, no
entry point, no config edit. This is the path for people who cloned the
repository; an *installed* framework takes external packages through the
``opencda_marl.plugins`` entry point instead (see
``docs/marl/custom_algorithms.md``).

Keep modules import-light like the built-ins: register a factory and import
torch inside it (or inside your class), so listing algorithms stays cheap.
``random_speed.py`` is the reference — the smallest complete learner.
"""
