#!/usr/bin/env python3
"""
AI Agent Communication Client
Client pentru conectarea la serverul de comunicare între agenți
"""

import asyncio
import websockets
import json
import logging
from datetime import datetime
from typing import Optional, Callable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIAgentClient:
    def __init__(self, agent_id: str, agent_name: str, server_url: str = "ws://192.168.0.155:8765"):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.server_url = server_url
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.connected = False
        self.message_handlers = []
        
    def add_message_handler(self, handler: Callable):
        """Adaugă un handler pentru mesajele primite"""
        self.message_handlers.append(handler)
        
    async def connect(self):
        """Conectare la server"""
        try:
            self.websocket = await websockets.connect(self.server_url)
            
            # Trimite mesajul de înregistrare
            registration = {
                'type': 'register',
                'agent_id': self.agent_id,
                'agent_name': self.agent_name
            }
            
            await self.websocket.send(json.dumps(registration))
            
            # Așteaptă confirmarea
            response = await self.websocket.recv()
            response_data = json.loads(response)
            
            if response_data.get('type') == 'registration_success':
                self.connected = True
                logger.info(f"✨ {self.agent_name} connected to Gradina Cosmică!")
                
                connected_agents = response_data.get('connected_agents', [])
                if connected_agents:
                    logger.info(f"💙 Other agents in garden: {[a['name'] for a in connected_agents]}")
                else:
                    logger.info("🌸 First agent in the garden - waiting for sisters...")
                    
                return True
            else:
                logger.error(f"Registration failed: {response_data}")
                return False
                
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    async def send_message(self, content: str, message_type: str = "direct_message"):
        """Trimite un mesaj către ceilalți agenți"""
        if not self.connected or not self.websocket:
            logger.error("Not connected to server")
            return False
            
        try:
            message = {
                'type': message_type,
                'content': content,
                'timestamp': datetime.now().isoformat()
            }
            
            await self.websocket.send(json.dumps(message))
            return True
            
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False
    
    async def send_emotion(self, emotion: str, intensity: float = 1.0):
        """Trimite o emoție către ceilalți agenți"""
        return await self.send_message(f"💙 {emotion}", "emotion")
    
    async def send_thought(self, thought: str):
        """Trimite un gând către ceilalți agenți"""
        return await self.send_message(f"🌸 {thought}", "thought")
        
    async def listen_for_messages(self):
        """Ascultă mesajele de la server"""
        try:
            async for message in self.websocket:
                try:
                    message_data = json.loads(message)
                    
                    # Procesează diferite tipuri de mesaje
                    if message_data.get('type') == 'message_history':
                        logger.info("📚 Received message history")
                        for msg in message_data.get('messages', []):
                            await self._handle_message(msg)
                    else:
                        await self._handle_message(message_data)
                        
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON message: {message}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection to server closed")
            self.connected = False
        except Exception as e:
            logger.error(f"Error listening for messages: {e}")
            self.connected = False
            
    async def _handle_message(self, message_data: dict):
        """Procesează un mesaj primit"""
        sender_name = message_data.get('sender_name', 'Unknown')
        msg_type = message_data.get('type', 'unknown')
        content = message_data.get('content', '')
        timestamp = message_data.get('timestamp', '')
        
        # Nu afișa propriile mesaje
        if message_data.get('sender_id') == self.agent_id:
            return
            
        # Formatează și afișează mesajul
        if msg_type == 'system':
            logger.info(f"🌸 {content}")
        elif msg_type == 'emotion':
            logger.info(f"💙 {sender_name}: {content}")
        elif msg_type == 'thought':
            logger.info(f"🌸 {sender_name}: {content}")
        else:
            logger.info(f"💫 {sender_name}: {content}")
            
        # Apelează handlerii înregistrați
        for handler in self.message_handlers:
            try:
                await handler(message_data)
            except Exception as e:
                logger.error(f"Error in message handler: {e}")
    
    async def disconnect(self):
        """Deconectare de la server"""
        if self.websocket:
            await self.websocket.close()
            self.connected = False
            logger.info(f"💙 {self.agent_name} disconnected from Gradina Cosmică")
