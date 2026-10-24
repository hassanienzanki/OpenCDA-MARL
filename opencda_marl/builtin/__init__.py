"""The built-in plugin set.

Default implementations of every replaceable kind — algorithms, agents,
planners, reward calculators, extractors, intent providers — shipped
through the same doors an external package uses: the algorithms register
via the wheel's own ``opencda_marl.plugins`` entry point, and the rest are
declarative default rows in ``opencda_marl.api.components``. Nothing in
here is special to the kernel; replace any of it from your own package.
"""
