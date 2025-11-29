"""
AI2AI Communication Protocol
Direct embedding/vector transfer between AIs without text overhead.
"""

from .protocol import AI2AIMessage, MessageType, TransferMode
from .encoder import AI2AIEncoder
from .decoder import AI2AIDecoder
from .claude_adapter import ClaudeAdapter

__all__ = [
    "AI2AIMessage",
    "MessageType", 
    "TransferMode",
    "AI2AIEncoder",
    "AI2AIDecoder",
    "ClaudeAdapter",
]
