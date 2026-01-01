"""
💾 NOVA Persistent Conversation Storage

Saves and loads conversation history to/from disk.
Ensures Nova remembers across server restarts.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ConversationStore:
    """
    Persistent storage for conversation history.
    Saves to JSON files, one per conversation session.
    """
    
    def __init__(self, storage_dir: str = "data/conversations"):
        """
        Initialize conversation store.
        
        Args:
            storage_dir: Directory to store conversation files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Current session file
        self.current_session_file = self.storage_dir / "current_session.json"
        
        logger.info(f"✓ Conversation store initialized")
        logger.info(f"  Storage: {self.storage_dir}")
    
    def save_conversation(self, messages: List[Dict], session_id: Optional[str] = None):
        """
        Save conversation to disk.
        
        Args:
            messages: List of message dictionaries
            session_id: Optional session identifier
        """
        try:
            # Prepare conversation data
            conversation = {
                'session_id': session_id or datetime.now().strftime("%Y%m%d_%H%M%S"),
                'timestamp': datetime.now().isoformat(),
                'message_count': len(messages),
                'messages': messages
            }
            
            # Save to current session file
            with open(self.current_session_file, 'w', encoding='utf-8') as f:
                json.dump(conversation, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"💾 Saved {len(messages)} messages to disk")
            
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")
    
    def load_conversation(self) -> List[Dict]:
        """
        Load most recent conversation from disk.
        
        Returns:
            List of message dictionaries
        """
        try:
            if not self.current_session_file.exists():
                logger.info("No previous conversation found")
                return []
            
            with open(self.current_session_file, 'r', encoding='utf-8') as f:
                conversation = json.load(f)
            
            messages = conversation.get('messages', [])
            logger.info(f"📂 Loaded {len(messages)} messages from previous session")
            logger.info(f"  Session: {conversation.get('session_id')}")
            logger.info(f"  Timestamp: {conversation.get('timestamp')}")
            
            return messages
            
        except Exception as e:
            logger.error(f"Failed to load conversation: {e}")
            return []
    
    def archive_conversation(self, session_id: Optional[str] = None):
        """
        Archive current conversation with timestamp.
        
        Args:
            session_id: Optional custom session ID
        """
        try:
            if not self.current_session_file.exists():
                return
            
            # Generate archive filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_name = f"conversation_{session_id or timestamp}.json"
            archive_path = self.storage_dir / archive_name
            
            # Move current to archive
            self.current_session_file.rename(archive_path)
            
            logger.info(f"📦 Archived conversation to {archive_name}")
            
        except Exception as e:
            logger.error(f"Failed to archive conversation: {e}")
    
    def list_conversations(self) -> List[Dict]:
        """
        List all archived conversations.
        
        Returns:
            List of conversation metadata
        """
        conversations = []
        
        for file in self.storage_dir.glob("conversation_*.json"):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    conversations.append({
                        'filename': file.name,
                        'session_id': data.get('session_id'),
                        'timestamp': data.get('timestamp'),
                        'message_count': data.get('message_count')
                    })
            except Exception as e:
                logger.warning(f"Could not read {file.name}: {e}")
        
        # Sort by timestamp (newest first)
        conversations.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return conversations
    
    def load_conversation_by_id(self, session_id: str) -> List[Dict]:
        """
        Load specific conversation by session ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of message dictionaries
        """
        for file in self.storage_dir.glob(f"conversation_{session_id}*.json"):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    conversation = json.load(f)
                    return conversation.get('messages', [])
            except Exception as e:
                logger.error(f"Failed to load conversation {session_id}: {e}")
        
        return []
    
    def clear_current_session(self):
        """Clear current session file."""
        try:
            if self.current_session_file.exists():
                self.current_session_file.unlink()
                logger.info("🗑️ Cleared current session")
        except Exception as e:
            logger.error(f"Failed to clear session: {e}")
    
    def get_stats(self) -> Dict:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with stats
        """
        conversations = self.list_conversations()
        
        total_messages = sum(c['message_count'] for c in conversations)
        
        return {
            'total_conversations': len(conversations),
            'total_messages': total_messages,
            'has_current_session': self.current_session_file.exists(),
            'storage_directory': str(self.storage_dir)
        }
