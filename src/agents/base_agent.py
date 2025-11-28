"""
Base Agent - Foundation for all NOVA agents

This module provides the abstract base class for all specialized agents:
- Common interface and lifecycle management
- LLM integration with conversation history
- Memory management (short-term and long-term)
- Tool/action framework
- Error handling and retry logic

Agent Architecture:

    User Input
         ↓
    ┌─────────────┐
    │ Base Agent  │
    ├─────────────┤
    │ • Memory    │ ← Conversation history
    │ • LLM       │ ← Claude/Mistral interface
    │ • Tools     │ ← Available actions
    │ • State     │ ← Agent state management
    └─────────────┘
         ↓
    Agent Response

Specialized Agents (inherit from BaseAgent):
- ResearchAgent: Information gathering, web search
- DocumentAgent: Document processing, Q&A
- VisionAgent: Image analysis, OCR
- AudioAgent: Speech-to-text, audio processing
- CodeAgent: Code generation, analysis

Usage:

    # Create custom agent
    class MyAgent(BaseAgent):
        def process(self, input_text: str) -> str:
            response = self.generate_response(
                prompt=f"Process this: {input_text}",
                temperature=0.7
            )
            return response
    
    # Use agent
    agent = MyAgent(name="my_agent")
    result = agent.run("Hello, agent!")
    print(agent.get_conversation_history())

Author: NOVA Development Team
Date: 28 November 2025
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

from src.config.settings import get_settings
from src.core.llm_interface import LLMInterface, LLMProvider


logger = logging.getLogger(__name__)


class AgentState(str, Enum):
    """Agent execution states"""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    WAITING = "waiting"
    ERROR = "error"
    COMPLETED = "completed"


@dataclass
class Message:
    """
    Conversation message
    
    Attributes:
        role: Message role (user, assistant, system)
        content: Message content
        timestamp: Message timestamp
        metadata: Additional metadata
    """
    role: str  # user, assistant, system
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"Message(role={self.role}, content={self.content[:50]}...)"


@dataclass
class AgentAction:
    """
    Agent action/tool invocation
    
    Attributes:
        name: Action name
        parameters: Action parameters
        result: Action result (after execution)
        error: Error message if failed
    """
    name: str
    parameters: Dict[str, Any]
    result: Optional[Any] = None
    error: Optional[str] = None
    
    def __repr__(self) -> str:
        status = "✓" if self.result else "✗" if self.error else "⏳"
        return f"Action({status} {self.name})"


class BaseAgent(ABC):
    """
    Abstract base class for all agents
    
    Provides:
    - LLM integration (Claude/Mistral)
    - Conversation memory management
    - Tool/action framework
    - State management
    - Error handling
    
    Subclasses must implement:
    - process(): Main agent logic
    - get_system_prompt(): Agent-specific instructions
    
    Args:
        name: Agent identifier
        llm_provider: LLM provider to use
        model: Model name (optional, uses default)
        temperature: Sampling temperature
        max_tokens: Maximum response tokens
        memory_size: Max messages to keep in memory
    
    Example:
        >>> agent = ResearchAgent(name="researcher")
        >>> result = agent.run("What is quantum computing?")
    """
    
    def __init__(
        self,
        name: str,
        llm_provider: Optional[LLMProvider] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        memory_size: int = 50
    ):
        """Initialize base agent"""
        self.name = name
        self.settings = get_settings()
        
        # LLM interface - initialize with provider
        llm_provider = llm_provider or LLMProvider.ANTHROPIC
        self.llm = LLMInterface(
            provider=llm_provider,
            model=model
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Memory
        self.memory_size = memory_size
        self.conversation_history: List[Message] = []
        self.actions_taken: List[AgentAction] = []
        
        # State
        self.state = AgentState.IDLE
        self._tools: Dict[str, Callable] = {}
        
        # Statistics
        self.total_calls = 0
        self.total_tokens = 0
        self.start_time = time.time()
        
        logger.info(f"Initialized {self.__class__.__name__}: {self.name}")
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Get agent-specific system prompt
        
        Returns:
            System prompt defining agent behavior
        """
        pass
    
    @abstractmethod
    def process(self, input_text: str, **kwargs) -> str:
        """
        Process user input (main agent logic)
        
        Args:
            input_text: User input
            **kwargs: Additional parameters
        
        Returns:
            Agent response
        """
        pass
    
    def run(
        self,
        input_text: str,
        add_to_history: bool = True,
        **kwargs
    ) -> str:
        """
        Run agent on input (high-level interface)
        
        Steps:
        1. Add user message to history
        2. Process input with agent logic
        3. Add assistant response to history
        4. Update statistics
        
        Args:
            input_text: User input
            add_to_history: Whether to save in memory
            **kwargs: Additional parameters
        
        Returns:
            Agent response
        """
        try:
            # Change state
            self.state = AgentState.THINKING
            
            # Add user message
            if add_to_history:
                self.add_message("user", input_text)
            
            # Process with agent logic
            response = self.process(input_text, **kwargs)
            
            # Add assistant response
            if add_to_history:
                self.add_message("assistant", response)
            
            # Update state
            self.state = AgentState.COMPLETED
            self.total_calls += 1
            
            return response
        
        except Exception as e:
            self.state = AgentState.ERROR
            logger.error(f"Agent {self.name} error: {e}", exc_info=True)
            raise
    
    def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_history: bool = False
    ) -> str:
        """
        Generate LLM response
        
        Args:
            prompt: User prompt
            system_prompt: Override system prompt
            temperature: Override temperature
            max_tokens: Override max tokens
            use_history: Include conversation history
        
        Returns:
            LLM response text
        """
        self.state = AgentState.THINKING
        
        # Build prompt with history if requested
        full_prompt = prompt
        
        if use_history and self.conversation_history:
            # Include recent conversation history
            history_text = []
            for msg in self.conversation_history[-10:]:  # Last 10 messages
                history_text.append(f"{msg.role.upper()}: {msg.content}")
            
            full_prompt = "\n\n".join(history_text) + f"\n\nUSER: {prompt}"
        
        # Generate with LLMInterface
        response = self.llm.generate(
            prompt=full_prompt,
            system=system_prompt or self.get_system_prompt(),
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens
        )
        
        # Update statistics
        self.total_tokens += response.usage.get('total_tokens', 0)
        
        return response.text
    
    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add message to conversation history
        
        Args:
            role: Message role (user/assistant/system)
            content: Message content
            metadata: Additional metadata
        """
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.conversation_history.append(message)
        
        # Trim memory if needed
        if len(self.conversation_history) > self.memory_size:
            # Keep system messages + recent messages
            system_messages = [m for m in self.conversation_history if m.role == "system"]
            recent_messages = [m for m in self.conversation_history if m.role != "system"]
            recent_messages = recent_messages[-(self.memory_size - len(system_messages)):]
            self.conversation_history = system_messages + recent_messages
    
    def get_conversation_history(
        self,
        n_messages: Optional[int] = None,
        role_filter: Optional[str] = None
    ) -> List[Message]:
        """
        Get conversation history
        
        Args:
            n_messages: Number of recent messages (None = all)
            role_filter: Filter by role (user/assistant/system)
        
        Returns:
            List of messages
        """
        messages = self.conversation_history
        
        # Filter by role
        if role_filter:
            messages = [m for m in messages if m.role == role_filter]
        
        # Limit count
        if n_messages:
            messages = messages[-n_messages:]
        
        return messages
    
    def clear_history(self) -> None:
        """Clear conversation history"""
        self.conversation_history.clear()
        logger.info(f"Cleared history for agent {self.name}")
    
    def register_tool(
        self,
        name: str,
        function: Callable,
        description: Optional[str] = None
    ) -> None:
        """
        Register a tool/action for the agent
        
        Args:
            name: Tool name
            function: Callable function
            description: Tool description
        """
        self._tools[name] = function
        logger.info(f"Registered tool '{name}' for agent {self.name}")
    
    def execute_action(
        self,
        action_name: str,
        parameters: Dict[str, Any]
    ) -> AgentAction:
        """
        Execute an action/tool
        
        Args:
            action_name: Action name
            parameters: Action parameters
        
        Returns:
            AgentAction with result or error
        """
        self.state = AgentState.ACTING
        
        action = AgentAction(name=action_name, parameters=parameters)
        
        try:
            # Check if tool exists
            if action_name not in self._tools:
                raise ValueError(f"Unknown tool: {action_name}")
            
            # Execute tool
            tool_func = self._tools[action_name]
            result = tool_func(**parameters)
            
            action.result = result
            logger.info(f"Action {action_name} succeeded")
        
        except Exception as e:
            action.error = str(e)
            logger.error(f"Action {action_name} failed: {e}")
        
        # Record action
        self.actions_taken.append(action)
        
        return action
    
    def get_available_tools(self) -> List[str]:
        """
        Get list of available tools
        
        Returns:
            List of tool names
        """
        return list(self._tools.keys())
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get agent statistics
        
        Returns:
            Statistics dictionary
        """
        uptime = time.time() - self.start_time
        
        return {
            'name': self.name,
            'state': self.state.value,
            'total_calls': self.total_calls,
            'total_tokens': self.total_tokens,
            'conversation_length': len(self.conversation_history),
            'actions_taken': len(self.actions_taken),
            'uptime_seconds': uptime,
            'tools_available': len(self._tools)
        }
    
    def reset(self) -> None:
        """Reset agent state"""
        self.clear_history()
        self.actions_taken.clear()
        self.state = AgentState.IDLE
        self.total_calls = 0
        self.total_tokens = 0
        logger.info(f"Reset agent {self.name}")
    
    def __repr__(self) -> str:
        """String representation"""
        return (
            f"{self.__class__.__name__}(name={self.name}, "
            f"state={self.state.value}, "
            f"messages={len(self.conversation_history)})"
        )


class SimpleAgent(BaseAgent):
    """
    Simple agent implementation for testing
    
    Just echoes input with LLM processing
    """
    
    def get_system_prompt(self) -> str:
        """System prompt for simple agent"""
        return (
            "You are a helpful AI assistant. "
            "Respond to user queries in a clear and concise manner. "
            "Be friendly and professional."
        )
    
    def process(self, input_text: str, **kwargs) -> str:
        """Process input with LLM"""
        return self.generate_response(
            prompt=input_text,
            use_history=kwargs.get('use_history', False)
        )


if __name__ == "__main__":
    """Test base agent"""
    print("=" * 80)
    print("Testing Base Agent")
    print("=" * 80)
    
    # Test 1: Create simple agent
    print("\n" + "-" * 80)
    print("Test 1: Create Simple Agent")
    print("-" * 80)
    
    agent = SimpleAgent(
        name="test_agent",
        temperature=0.7,
        max_tokens=500
    )
    
    print(f"✅ Created agent: {agent}")
    print(f"   State: {agent.state.value}")
    print(f"   System prompt: {agent.get_system_prompt()[:80]}...")
    
    # Test 2: Run agent (simple echo)
    print("\n" + "-" * 80)
    print("Test 2: Run Agent with Input")
    print("-" * 80)
    
    try:
        response = agent.run(
            "What is 2+2? Answer in one sentence.",
            use_history=False
        )
        print(f"✅ Agent response: {response[:200]}...")
        print(f"   State: {agent.state.value}")
        print(f"   History length: {len(agent.conversation_history)}")
    except Exception as e:
        print(f"⚠️  Agent run failed (expected if no API keys): {e}")
    
    # Test 3: Memory management
    print("\n" + "-" * 80)
    print("Test 3: Memory Management")
    print("-" * 80)
    
    agent.add_message("user", "Hello")
    agent.add_message("assistant", "Hi there!")
    agent.add_message("user", "How are you?")
    agent.add_message("assistant", "I'm doing well, thanks!")
    
    print(f"✅ Added 4 messages to history")
    print(f"   Total messages: {len(agent.conversation_history)}")
    
    recent = agent.get_conversation_history(n_messages=2)
    print(f"   Recent 2 messages: {len(recent)}")
    
    user_messages = agent.get_conversation_history(role_filter="user")
    print(f"   User messages: {len(user_messages)}")
    
    # Test 4: Tool registration
    print("\n" + "-" * 80)
    print("Test 4: Tool Registration & Execution")
    print("-" * 80)
    
    def add_numbers(a: int, b: int) -> int:
        """Test tool: Add two numbers"""
        return a + b
    
    def multiply_numbers(a: int, b: int) -> int:
        """Test tool: Multiply two numbers"""
        return a * b
    
    agent.register_tool("add", add_numbers, "Add two numbers")
    agent.register_tool("multiply", multiply_numbers, "Multiply two numbers")
    
    print(f"✅ Registered 2 tools")
    print(f"   Available tools: {agent.get_available_tools()}")
    
    # Execute actions
    action1 = agent.execute_action("add", {"a": 5, "b": 3})
    print(f"   Action 1 (add): {action1} → {action1.result}")
    
    action2 = agent.execute_action("multiply", {"a": 4, "b": 7})
    print(f"   Action 2 (multiply): {action2} → {action2.result}")
    
    # Test 5: Statistics
    print("\n" + "-" * 80)
    print("Test 5: Agent Statistics")
    print("-" * 80)
    
    stats = agent.get_statistics()
    print(f"✅ Agent statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Test 6: Reset
    print("\n" + "-" * 80)
    print("Test 6: Reset Agent")
    print("-" * 80)
    
    agent.reset()
    print(f"✅ Agent reset")
    print(f"   State: {agent.state.value}")
    print(f"   History length: {len(agent.conversation_history)}")
    print(f"   Actions: {len(agent.actions_taken)}")
    
    # Test 7: Memory size limit
    print("\n" + "-" * 80)
    print("Test 7: Memory Size Limit")
    print("-" * 80)
    
    small_agent = SimpleAgent(name="small_memory", memory_size=5)
    
    # Add 10 messages
    for i in range(10):
        small_agent.add_message("user", f"Message {i}")
    
    print(f"✅ Added 10 messages to agent with memory_size=5")
    print(f"   Actual history length: {len(small_agent.conversation_history)}")
    print(f"   Memory correctly trimmed: {len(small_agent.conversation_history) <= 5}")
    
    print("\n" + "=" * 80)
    print("Base Agent tests completed")
    print("=" * 80)
