"""An advanced, web-connected agent built on the Claude API."""

from .agent import Agent
from .config import AgentConfig
from .tools import ToolRegistry

__all__ = ["Agent", "AgentConfig", "ToolRegistry"]
__version__ = "0.1.0"
