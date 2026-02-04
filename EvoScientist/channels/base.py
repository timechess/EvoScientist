"""Abstract base class for communication channels.

This module defines the Channel interface that all messaging channels
(iMessage, WeChat, etc.) must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncIterator


@dataclass
class IncomingMessage:
    """Represents a message received from a channel."""

    sender: str  # Phone number, email, or unique identifier
    content: str  # Message text content
    timestamp: datetime  # When the message was sent
    message_id: str  # Unique identifier for the message
    metadata: dict = field(default_factory=dict)  # Channel-specific metadata


@dataclass
class OutgoingMessage:
    """Represents a message to be sent through a channel."""

    recipient: str  # Phone number, email, or unique identifier
    content: str  # Message text content
    reply_to: str | None = None  # Optional message ID being replied to
    metadata: dict = field(default_factory=dict)  # Channel-specific metadata


class Channel(ABC):
    """Abstract base class for messaging channels.

    Subclasses must implement:
    - start(): Initialize the channel (connect, authenticate, etc.)
    - stop(): Clean up resources
    - receive(): Async iterator yielding incoming messages
    - send(): Send a message through the channel
    """

    @abstractmethod
    async def start(self) -> None:
        """Initialize and start the channel.

        This method should:
        - Establish connections
        - Verify permissions/authentication
        - Start any background tasks needed

        Raises:
            ChannelError: If initialization fails
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the channel and clean up resources.

        This method should:
        - Close connections
        - Cancel background tasks
        - Release any held resources
        """
        pass

    @abstractmethod
    async def receive(self) -> AsyncIterator[IncomingMessage]:
        """Async iterator that yields incoming messages.

        Yields:
            IncomingMessage: Each new message received

        Example:
            async for msg in channel.receive():
                print(f"From {msg.sender}: {msg.content}")
        """
        pass

    @abstractmethod
    async def send(self, message: OutgoingMessage) -> bool:
        """Send a message through the channel.

        Args:
            message: The message to send

        Returns:
            True if sent successfully, False otherwise
        """
        pass


class ChannelError(Exception):
    """Base exception for channel-related errors."""

    pass


class ChannelPermissionError(ChannelError):
    """Raised when the channel lacks required permissions."""

    pass


class ChannelConnectionError(ChannelError):
    """Raised when the channel cannot establish a connection."""

    pass
