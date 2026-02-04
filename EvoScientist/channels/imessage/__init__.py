"""iMessage channel implementation for EvoScientist.

Uses imsg CLI via JSON-RPC for real-time message streaming.

Requirements:
- macOS only
- imsg CLI: brew install steipete/tap/imsg
- Full Disk Access permission
- Messages.app logged into iCloud
"""

from .channel_rpc import IMessageChannelRpc as IMessageChannel
from .channel_rpc import IMessageConfig

__all__ = ["IMessageChannel", "IMessageConfig"]
