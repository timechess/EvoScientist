"""Communication channels for EvoScientist.

This module provides an extensible interface for different messaging channels
(iMessage, WeChat, etc.) to communicate with the EvoScientist agent.
"""

from .base import Channel, IncomingMessage, OutgoingMessage

__all__ = ["Channel", "IncomingMessage", "OutgoingMessage"]
