"""Agent system registry: maps config strings to AgentSystem subclasses.

Usage::

    from agents.registry import create_agent_system

    agent = create_agent_system(agent_config)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Type

from agents.base import AgentSystem
from models.config import AgentConfig

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Registry mapping
# ---------------------------------------------------------------------------
_REGISTRY: dict[str, Type[AgentSystem]] = {}


def register(name: str):
    """Decorator to register an ``AgentSystem`` subclass under *name*."""

    def _decorator(cls: Type[AgentSystem]) -> Type[AgentSystem]:
        if name in _REGISTRY:
            raise ValueError(f"Agent system '{name}' is already registered.")
        _REGISTRY[name] = cls
        return cls

    return _decorator


def create_agent_system(config: AgentConfig) -> AgentSystem:
    """Instantiate the agent system specified in *config*.

    Raises ``KeyError`` if ``config.agent_system`` is not registered.
    """
    # Lazy-import concrete implementations so they self-register.
    _ensure_builtins_loaded()

    key = config.agent_system
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(
            f"Unknown agent system '{key}'. Available: {available}."
        )
    return _REGISTRY[key](config)


def _ensure_builtins_loaded() -> None:
    """Import built-in agent modules so their ``@register`` calls execute."""
    # Each import triggers the @register decorator at module level.
    import agents.single_llm  # noqa: F401
