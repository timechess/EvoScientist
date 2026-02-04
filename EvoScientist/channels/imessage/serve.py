"""iMessage channel server.

Standalone script to run the iMessage channel with CLI options.

Usage:
    python -m EvoScientist.channels.imessage.serve [OPTIONS]

Examples:
    # Allow all senders (default)
    python -m EvoScientist.channels.imessage.serve

    # Only allow specific senders
    python -m EvoScientist.channels.imessage.serve --allow +1234567890 --allow user@example.com

    # Custom imsg path
    python -m EvoScientist.channels.imessage.serve --cli-path /usr/local/bin/imsg
"""

import asyncio
import argparse
import logging
import signal
import sys
from typing import Callable

from . import IMessageChannel, IMessageConfig
from ..base import OutgoingMessage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_agent_handler():
    """Create handler that uses EvoScientist agent."""
    from langchain_core.messages import HumanMessage
    from ...EvoScientist import create_cli_agent

    agent = create_cli_agent()
    sessions: dict[str, str] = {}  # sender -> thread_id

    async def handler(msg) -> str:
        import uuid
        sender = msg.sender
        if sender not in sessions:
            sessions[sender] = str(uuid.uuid4())
        thread_id = sessions[sender]

        config = {"configurable": {"thread_id": thread_id}}
        result = agent.invoke(
            {"messages": [HumanMessage(content=msg.content)]},
            config=config,
        )
        # Extract last AI message
        messages = result.get("messages", [])
        for m in reversed(messages):
            if hasattr(m, "content") and m.type == "ai":
                return m.content
        return "No response"

    return handler


class IMessageServer:
    """Server that runs the iMessage channel and handles messages."""

    def __init__(
        self,
        config: IMessageConfig,
        handler: Callable | None = None,
    ):
        self.config = config
        self.channel = IMessageChannel(config)
        self.handler = handler or self._default_handler
        self._running = False

    async def _default_handler(self, msg) -> str:
        """Default echo handler."""
        return f"Echo: {msg.content}"

    async def run(self) -> None:
        """Run the server."""
        await self.channel.start()
        self._running = True

        logger.info("iMessage server running. Press Ctrl+C to stop.")
        if self.config.allowed_senders:
            logger.info(f"Allowed senders: {self.config.allowed_senders}")
        else:
            logger.info("Allowing all senders")

        try:
            async for msg in self.channel.receive():
                logger.info(f"From {msg.sender}: {msg.content[:50]}...")
                try:
                    response = await self.handler(msg)
                    if response:
                        await self.channel.send(OutgoingMessage(
                            recipient=msg.sender,
                            content=response,
                            metadata=msg.metadata,
                        ))
                except Exception as e:
                    logger.error(f"Handler error: {e}")
        finally:
            await self.channel.stop()

    async def stop(self) -> None:
        """Stop the server."""
        self._running = False
        await self.channel.stop()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="iMessage channel server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--allow",
        action="append",
        dest="allowed_senders",
        help="Allowed sender (phone/email). Can be used multiple times.",
    )
    parser.add_argument(
        "--cli-path",
        default="imsg",
        help="Path to imsg CLI (default: imsg)",
    )
    parser.add_argument(
        "--db-path",
        help="Path to Messages database",
    )
    parser.add_argument(
        "--attachments",
        action="store_true",
        help="Include attachments in messages",
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Use EvoScientist agent as handler (default: echo)",
    )
    return parser.parse_args()


async def async_main():
    """Async entry point."""
    args = parse_args()

    config = IMessageConfig(
        cli_path=args.cli_path,
        db_path=args.db_path,
        allowed_senders=set(args.allowed_senders) if args.allowed_senders else None,
        include_attachments=args.attachments,
    )

    handler = None
    if args.agent:
        logger.info("Loading EvoScientist agent...")
        handler = create_agent_handler()
        logger.info("Agent loaded")

    server = IMessageServer(config, handler=handler)

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(server.stop()))

    await server.run()


def main():
    """Entry point."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
