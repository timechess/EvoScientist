"""iMessage channel using imsg JSON-RPC.

This is an improved implementation that uses the imsg CLI
via JSON-RPC, similar to OpenClaw's approach.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncIterator, Set

from ..base import Channel, IncomingMessage, OutgoingMessage, ChannelError
from .rpc_client import ImsgRpcClient, RpcNotification

logger = logging.getLogger(__name__)


@dataclass
class IMessageConfig:
    """Configuration for iMessage channel."""

    cli_path: str = "imsg"
    db_path: str | None = None
    allowed_senders: Set[str] | None = None
    include_attachments: bool = False
    text_chunk_limit: int = 4000
    service: str = "auto"  # imessage, sms, or auto


class IMessageChannelRpc(Channel):
    """iMessage channel using imsg JSON-RPC.

    This implementation uses the imsg CLI via JSON-RPC over stdio,
    providing real-time message streaming instead of polling.

    Args:
        config: Channel configuration
    """

    def __init__(self, config: IMessageConfig | None = None):
        self.config = config or IMessageConfig()
        self._client: ImsgRpcClient | None = None
        self._running = False
        self._message_queue: asyncio.Queue[IncomingMessage] = asyncio.Queue()
        self._subscription_id: int | None = None

    def _handle_notification(self, notification: RpcNotification) -> None:
        """Handle incoming RPC notifications."""
        if notification.method == "message":
            self._handle_message(notification.params)
        elif notification.method == "error":
            logger.error(f"imsg error: {notification.params}")

    def _handle_message(self, params: dict | None) -> None:
        """Process incoming message notification."""
        if not params:
            return

        message = params.get("message", {})
        if not message:
            return

        # Skip messages from self
        if message.get("is_from_me"):
            return

        sender = message.get("sender", "").strip()
        if not sender:
            return

        # Check allowed senders
        if not self._is_sender_allowed(sender):
            logger.debug(f"Ignoring message from {sender}")
            return

        text = message.get("text", "").strip()
        if not text:
            return

        # Parse timestamp
        timestamp = datetime.now()
        if created_at := message.get("created_at"):
            try:
                timestamp = datetime.fromisoformat(created_at)
            except ValueError:
                pass

        # Build metadata
        metadata = {
            "chat_id": message.get("chat_id"),
            "chat_guid": message.get("chat_guid"),
            "is_group": message.get("is_group", False),
            "chat_name": message.get("chat_name"),
        }

        # Handle attachments if enabled
        if self.config.include_attachments:
            attachments = message.get("attachments", [])
            if attachments:
                metadata["attachments"] = attachments

        incoming = IncomingMessage(
            sender=sender,
            content=text,
            timestamp=timestamp,
            message_id=str(message.get("id", "")),
            metadata=metadata,
        )

        try:
            self._message_queue.put_nowait(incoming)
        except asyncio.QueueFull:
            logger.warning("Message queue full, dropping message")

    def _is_sender_allowed(self, sender: str) -> bool:
        """Check if sender is in allowed list."""
        if not self.config.allowed_senders:
            return True
        return sender in self.config.allowed_senders

    def add_allowed_sender(self, sender: str) -> None:
        """Add a sender to the allowed list."""
        if self.config.allowed_senders is None:
            self.config.allowed_senders = set()
        self.config.allowed_senders.add(sender)
        logger.info(f"Added allowed sender: {sender}")

    def remove_allowed_sender(self, sender: str) -> None:
        """Remove a sender from the allowed list."""
        if self.config.allowed_senders:
            self.config.allowed_senders.discard(sender)
            logger.info(f"Removed allowed sender: {sender}")

    def clear_allowed_senders(self) -> None:
        """Clear allowed list (allow all)."""
        self.config.allowed_senders = None
        logger.info("Cleared allowed senders (allowing all)")

    def list_allowed_senders(self) -> set[str] | None:
        """Get current allowed senders."""
        return self.config.allowed_senders

    async def start(self) -> None:
        """Initialize and start the channel."""
        logger.info("Starting iMessage channel (RPC)...")

        self._client = ImsgRpcClient(
            cli_path=self.config.cli_path,
            db_path=self.config.db_path,
            on_notification=self._handle_notification,
        )

        try:
            await self._client.start()
        except Exception as e:
            raise ChannelError(f"Failed to start imsg: {e}") from e

        # Subscribe to message events
        try:
            result = await self._client.request(
                "watch.subscribe",
                {"attachments": self.config.include_attachments},
            )
            self._subscription_id = result.get("subscription")
        except Exception as e:
            await self._client.stop()
            raise ChannelError(f"Failed to subscribe: {e}") from e

        self._running = True
        logger.info("iMessage channel started")

    async def stop(self) -> None:
        """Stop the channel and clean up."""
        logger.info("Stopping iMessage channel...")
        self._running = False

        if self._client and self._subscription_id:
            try:
                await self._client.request(
                    "watch.unsubscribe",
                    {"subscription": self._subscription_id},
                )
            except Exception:
                pass

        if self._client:
            await self._client.stop()
            self._client = None

        logger.info("iMessage channel stopped")

    async def receive(self) -> AsyncIterator[IncomingMessage]:
        """Yield incoming messages from the queue."""
        while self._running:
            try:
                msg = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=1.0,
                )
                yield msg
            except asyncio.TimeoutError:
                continue

    def _segment_message(self, content: str) -> list[str]:
        """Split long message into segments."""
        limit = self.config.text_chunk_limit
        if len(content) <= limit:
            return [content]

        segments = []
        remaining = content

        while remaining:
            if len(remaining) <= limit:
                segments.append(remaining)
                break

            chunk = remaining[:limit]
            # Try split at newline
            nl_pos = chunk.rfind("\n")
            if nl_pos > limit // 2:
                split_pos = nl_pos + 1
            else:
                # Try split at space
                sp_pos = chunk.rfind(" ")
                if sp_pos > limit // 2:
                    split_pos = sp_pos + 1
                else:
                    split_pos = limit

            segments.append(remaining[:split_pos].rstrip())
            remaining = remaining[split_pos:].lstrip()

        return segments

    async def send(self, message: OutgoingMessage) -> bool:
        """Send a message via iMessage."""
        if not self._client:
            logger.error("Cannot send: client not running")
            return False

        segments = self._segment_message(message.content)

        for segment in segments:
            # Prefer chat_id over recipient (to) for replies
            chat_id = message.metadata.get("chat_id")

            if chat_id:
                params = {
                    "chat_id": chat_id,
                    "text": segment,
                    "service": self.config.service,
                }
            elif message.recipient:
                params = {
                    "to": message.recipient,
                    "text": segment,
                    "service": self.config.service,
                }
            else:
                logger.error("Cannot send: no recipient or chat_id")
                return False

            try:
                await self._client.request("send", params)
            except Exception as e:
                logger.error(f"Send failed: {e}")
                return False

        return True
