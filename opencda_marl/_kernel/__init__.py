"""The kernel: the complete lifecycle. Sealed.

Everything under this package is the framework's fixed machinery — the
coordinator, both backend environments and their episode loops, world and
scenario management, vehicle adapters, traffic orchestration, sensing and
the policy-driving machinery. External code does not import from here (the
leading underscore is the contract), and nothing in here imports a
concrete plugin: the kernel reaches every replaceable part through
``opencda_marl.api`` and the plugin registry, which is what keeps every
part of the loop replaceable while the loop itself stays stable.
An architecture test enforces both directions.
"""
