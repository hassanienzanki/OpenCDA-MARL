"""Built-in vehicle agent types and their factory helpers."""

__all__ = ["AgentFactory"]


def __getattr__(name):
    if name == "AgentFactory":
        from opencda_marl.builtin.agents.agent_factory import AgentFactory
        return AgentFactory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
