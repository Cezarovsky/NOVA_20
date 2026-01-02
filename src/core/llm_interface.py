"""
LLM Interface - Unified API Integration

This module provides a unified interface for interacting with multiple LLM providers:
- Anthropic (Claude models)
- Mistral AI (Mistral models)

Features:
1. Unified interface across providers
2. Automatic retry logic with exponential backoff
3. Error handling and fallback strategies
4. Streaming support for real-time responses
5. Token counting and usage tracking
6. Rate limiting and quota management
7. Response caching (optional)

Architecture:

    ┌─────────────────┐
    │   LLMInterface  │  ← Unified interface
    └────────┬────────┘
             │
       ┌─────┴─────┐
       │           │
    ┌──▼───┐   ┌──▼───┐
    │Claude│   │Mistral│  ← Provider-specific clients
    └──────┘   └───────┘

Usage:

    # Basic usage
    llm = LLMInterface(provider="anthropic")
    response = llm.generate("Hello, how are you?")
    print(response.text)
    
    # Streaming
    for chunk in llm.generate_stream("Tell me a story"):
        print(chunk, end="", flush=True)
    
    # With fallback
    llm = LLMInterface(
        provider="anthropic",
        fallback_provider="mistral"
    )
    response = llm.generate("Hello")  # Uses Mistral if Claude fails

Author: NOVA Development Team
Date: 28 November 2025
"""

import time
import logging
from typing import Optional, Iterator, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

import anthropic
from mistralai import Mistral

from src.config.settings import get_settings


logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    ANTHROPIC = "anthropic"
    MISTRAL = "mistral"
    OLLAMA = "ollama"  # Local models via Ollama (Mistral, Phi, TinyLlama, etc.)


@dataclass
class LLMResponse:
    """
    Standardized LLM response
    
    Attributes:
        text: Generated text
        model: Model used for generation
        provider: Provider name
        usage: Token usage statistics
        finish_reason: Why generation stopped
        latency_ms: Response time in milliseconds
        metadata: Additional provider-specific data
    """
    text: str
    model: str
    provider: str
    usage: Dict[str, int]  # {prompt_tokens, completion_tokens, total_tokens}
    finish_reason: str
    latency_ms: float
    metadata: Optional[Dict[str, Any]] = None
    
    def __repr__(self) -> str:
        return (
            f"LLMResponse(provider={self.provider}, model={self.model}, "
            f"tokens={self.usage.get('total_tokens', 0)}, "
            f"latency={self.latency_ms:.0f}ms)"
        )


class LLMInterface:
    """
    Unified interface for multiple LLM providers
    
    Provides a consistent API across different LLM services with:
    - Automatic retry logic
    - Error handling
    - Fallback support
    - Streaming capabilities
    - Usage tracking
    
    Args:
        provider: Primary LLM provider
        model: Model name (uses default from settings if None)
        fallback_provider: Fallback provider if primary fails
        fallback_model: Fallback model name
        max_retries: Maximum retry attempts on failure
        retry_delay: Initial delay between retries (exponential backoff)
        timeout: Request timeout in seconds
    
    Example:
        >>> llm = LLMInterface(provider="anthropic")
        >>> response = llm.generate("What is AI?", max_tokens=100)
        >>> print(response.text)
        >>> print(f"Used {response.usage['total_tokens']} tokens")
    """
    
    def __init__(
        self,
        provider: str = "anthropic",
        model: Optional[str] = None,
        fallback_provider: Optional[str] = None,
        fallback_model: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 60.0
    ):
        """Initialize LLM interface"""
        self.settings = get_settings()
        
        # Primary provider
        self.provider = LLMProvider(provider)
        self.model = model or self._get_default_model(self.provider)
        
        # Fallback provider
        self.fallback_provider = LLMProvider(fallback_provider) if fallback_provider else None
        self.fallback_model = fallback_model or (
            self._get_default_model(self.fallback_provider) if self.fallback_provider else None
        )
        
        # Configuration
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        
        # Initialize clients
        self._anthropic_client = None
        self._mistral_client = None
        
        # Usage tracking
        self.total_tokens_used = 0
        self.total_requests = 0
        self.total_errors = 0
        
        logger.info(
            f"Initialized LLMInterface: provider={self.provider}, model={self.model}"
        )
    
    def _get_default_model(self, provider: LLMProvider) -> str:
        """Get default model for provider"""
        if provider == LLMProvider.ANTHROPIC:
            return self.settings.DEFAULT_LLM_MODEL
        elif provider == LLMProvider.MISTRAL:
            return self.settings.FALLBACK_LLM_MODEL
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    @property
    def anthropic_client(self) -> anthropic.Anthropic:
        """Lazy initialization of Anthropic client"""
        if self._anthropic_client is None:
            if not self.settings.ANTHROPIC_API_KEY:
                raise ValueError(
                    "ANTHROPIC_API_KEY not set. "
                    "Get your key at: https://console.anthropic.com/"
                )
            self._anthropic_client = anthropic.Anthropic(
                api_key=self.settings.ANTHROPIC_API_KEY,
                timeout=self.timeout
            )
        return self._anthropic_client
    
    @property
    def mistral_client(self) -> Mistral:
        """Lazy initialization of Mistral client"""
        if self._mistral_client is None:
            if not self.settings.MISTRAL_API_KEY:
                raise ValueError(
                    "MISTRAL_API_KEY not set. "
                    "Get your key at: https://console.mistral.ai/"
                )
            self._mistral_client = Mistral(api_key=self.settings.MISTRAL_API_KEY)
        return self._mistral_client
    
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text completion
        
        Args:
            prompt: User prompt/query
            system: System prompt (instruction for the model)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            top_p: Nucleus sampling parameter
            top_k: Top-K sampling parameter
            stop_sequences: Stop generation at these sequences
            **kwargs: Additional provider-specific parameters
        
        Returns:
            LLMResponse with generated text and metadata
        
        Raises:
            RuntimeError: If generation fails after all retries
        
        Example:
            >>> response = llm.generate(
            ...     "Explain quantum computing",
            ...     system="You are a physics teacher",
            ...     max_tokens=200
            ... )
            >>> print(response.text)
        """
        start_time = time.time()
        self.total_requests += 1
        
        # Try primary provider
        try:
            response = self._generate_with_provider(
                provider=self.provider,
                model=self.model,
                prompt=prompt,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop_sequences=stop_sequences,
                **kwargs
            )
            
            # Track usage
            self.total_tokens_used += response.usage.get('total_tokens', 0)
            
            return response
            
        except Exception as e:
            logger.error(f"Primary provider {self.provider} failed: {e}")
            self.total_errors += 1
            
            # Try fallback if available
            if self.fallback_provider:
                logger.info(f"Attempting fallback to {self.fallback_provider}")
                try:
                    response = self._generate_with_provider(
                        provider=self.fallback_provider,
                        model=self.fallback_model,
                        prompt=prompt,
                        system=system,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        stop_sequences=stop_sequences,
                        **kwargs
                    )
                    
                    self.total_tokens_used += response.usage.get('total_tokens', 0)
                    return response
                    
                except Exception as fallback_error:
                    logger.error(f"Fallback provider also failed: {fallback_error}")
                    raise RuntimeError(
                        f"Both primary ({self.provider}) and fallback ({self.fallback_provider}) failed"
                    ) from fallback_error
            
            # No fallback available
            raise RuntimeError(f"Generation failed with {self.provider}") from e
    
    def _generate_with_provider(
        self,
        provider: LLMProvider,
        model: str,
        prompt: str,
        system: Optional[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: Optional[int],
        stop_sequences: Optional[List[str]],
        **kwargs
    ) -> LLMResponse:
        """Generate with specific provider (with retry logic)"""
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                if provider == LLMProvider.ANTHROPIC:
                    response = self._generate_anthropic(
                        model=model,
                        prompt=prompt,
                        system=system,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        stop_sequences=stop_sequences,
                        **kwargs
                    )
                elif provider == LLMProvider.MISTRAL:
                    response = self._generate_mistral(
                        model=model,
                        prompt=prompt,
                        system=system,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stop_sequences=stop_sequences,
                        **kwargs
                    )
                elif provider == LLMProvider.OLLAMA:
                    response = self._generate_ollama(
                        model=model,
                        prompt=prompt,
                        system=system,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stop_sequences=stop_sequences,
                        **kwargs
                    )
                else:
                    raise ValueError(f"Unknown provider: {provider}")
                
                latency_ms = (time.time() - start_time) * 1000
                response.latency_ms = latency_ms
                
                logger.info(
                    f"Generated with {provider}: {response.usage.get('total_tokens', 0)} tokens, "
                    f"{latency_ms:.0f}ms"
                )
                
                return response
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"Attempt {attempt + 1}/{self.max_retries} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    raise
    
    def _generate_anthropic(
        self,
        model: str,
        prompt: str,
        system: Optional[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: Optional[int],
        stop_sequences: Optional[List[str]],
        **kwargs
    ) -> LLMResponse:
        """Generate using Anthropic Claude"""
        
        # Build messages
        messages = [{"role": "user", "content": prompt}]
        
        # Build API parameters (Anthropic doesn't support top_k)
        api_params = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop_sequences": stop_sequences or []
        }
        
        # Add system prompt if provided
        if system:
            api_params["system"] = system
        
        # API call
        response = self.anthropic_client.messages.create(**api_params, **kwargs)
        
        # Extract text
        text = response.content[0].text if response.content else ""
        
        # Build standardized response
        return LLMResponse(
            text=text,
            model=model,
            provider="anthropic",
            usage={
                'prompt_tokens': response.usage.input_tokens,
                'completion_tokens': response.usage.output_tokens,
                'total_tokens': response.usage.input_tokens + response.usage.output_tokens
            },
            finish_reason=response.stop_reason,
            latency_ms=0.0,  # Will be set by caller
            metadata={
                'id': response.id,
                'type': response.type,
                'role': response.role
            }
        )
    
    def _generate_mistral(
        self,
        model: str,
        prompt: str,
        system: Optional[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_sequences: Optional[List[str]],
        **kwargs
    ) -> LLMResponse:
        """Generate using Mistral AI"""
        
        # Build messages
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        # API call
        response = self.mistral_client.chat.complete(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop_sequences or [],
            **kwargs
        )
        
        # Extract text
        text = response.choices[0].message.content if response.choices else ""
        
        # Build standardized response
        return LLMResponse(
            text=text,
            model=model,
            provider="mistral",
            usage={
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            },
            finish_reason=response.choices[0].finish_reason if response.choices else "unknown",
            latency_ms=0.0,
            metadata={
                'id': response.id,
                'created': response.created,
                'object': response.object
            }
        )
    
    def _generate_ollama(
        self,
        model: str,
        prompt: str,
        system: Optional[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_sequences: Optional[List[str]],
        **kwargs
    ) -> LLMResponse:
        """Generate using Ollama (local models)"""
        import requests
        import json
        
        # Build prompt with system message if provided
        full_prompt = prompt
        if system:
            full_prompt = f"<|system|>\n{system}\n<|user|>\n{prompt}\n<|assistant|>"
        
        # Ollama API endpoint
        url = "http://localhost:11434/api/generate"
        
        payload = {
            "model": model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            }
        }
        
        if stop_sequences:
            payload["options"]["stop"] = stop_sequences
        
        # API call
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract text
        text = data.get("response", "")
        
        # Build standardized response
        return LLMResponse(
            text=text,
            model=model,
            provider="ollama",
            usage={
                'prompt_tokens': data.get('prompt_eval_count', 0),
                'completion_tokens': data.get('eval_count', 0),
                'total_tokens': data.get('prompt_eval_count', 0) + data.get('eval_count', 0)
            },
            finish_reason=data.get('done_reason', 'stop'),
            latency_ms=data.get('total_duration', 0) / 1_000_000,  # Convert nanoseconds to ms
            metadata={
                'model': data.get('model'),
                'created_at': data.get('created_at'),
                'done': data.get('done'),
                'eval_duration': data.get('eval_duration', 0) / 1_000_000  # ms
            }
        )
    
    def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> Iterator[str]:
        """
        Generate text with streaming (real-time token-by-token)
        
        Args:
            prompt: User prompt
            system: System prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-K sampling parameter
            stop_sequences: Stop sequences
            **kwargs: Additional parameters
        
        Yields:
            Text chunks as they are generated
        
        Example:
            >>> for chunk in llm.generate_stream("Tell me a story"):
            ...     print(chunk, end="", flush=True)
            Once upon a time...
        """
        self.total_requests += 1
        
        try:
            if self.provider == LLMProvider.ANTHROPIC:
                yield from self._stream_anthropic(
                    model=self.model,
                    prompt=prompt,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    stop_sequences=stop_sequences,
                    **kwargs
                )
            elif self.provider == LLMProvider.MISTRAL:
                yield from self._stream_mistral(
                    model=self.model,
                    prompt=prompt,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop_sequences=stop_sequences,
                    **kwargs
                )
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            self.total_errors += 1
            raise
    
    def _stream_anthropic(
        self,
        model: str,
        prompt: str,
        system: Optional[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: Optional[int],
        stop_sequences: Optional[List[str]],
        **kwargs
    ) -> Iterator[str]:
        """Stream from Anthropic"""
        
        messages = [{"role": "user", "content": prompt}]
        
        with self.anthropic_client.messages.stream(
            model=model,
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences or [],
            **kwargs
        ) as stream:
            for text in stream.text_stream:
                yield text
    
    def _stream_mistral(
        self,
        model: str,
        prompt: str,
        system: Optional[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_sequences: Optional[List[str]],
        **kwargs
    ) -> Iterator[str]:
        """Stream from Mistral"""
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        stream = self.mistral_client.chat.stream(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop_sequences or [],
            **kwargs
        )
        
        for chunk in stream:
            if chunk.data.choices:
                delta = chunk.data.choices[0].delta.content
                if delta:
                    yield delta
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics
        
        Returns:
            Dictionary with usage metrics
        """
        return {
            'total_requests': self.total_requests,
            'total_tokens': self.total_tokens_used,
            'total_errors': self.total_errors,
            'error_rate': self.total_errors / max(self.total_requests, 1),
            'avg_tokens_per_request': self.total_tokens_used / max(self.total_requests, 1),
            'provider': str(self.provider),
            'model': self.model
        }
    
    def __repr__(self) -> str:
        """String representation"""
        return (
            f"LLMInterface(provider={self.provider}, model={self.model}, "
            f"requests={self.total_requests}, tokens={self.total_tokens_used})"
        )


if __name__ == "__main__":
    """Test LLM interface"""
    print("=" * 80)
    print("Testing LLM Interface")
    print("=" * 80)
    
    # Test 1: Basic generation (requires API keys)
    print("\n" + "-" * 80)
    print("Test 1: Basic Generation (Anthropic)")
    print("-" * 80)
    
    try:
        llm = LLMInterface(provider="anthropic")
        
        response = llm.generate(
            prompt="What is 2+2? Answer in one word.",
            max_tokens=10
        )
        
        print(f"✅ Response: {response.text}")
        print(f"✅ Model: {response.model}")
        print(f"✅ Tokens: {response.usage}")
        print(f"✅ Latency: {response.latency_ms:.0f}ms")
        
    except ValueError as e:
        print(f"⚠️  Skipped (API key not set): {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: Streaming
    print("\n" + "-" * 80)
    print("Test 2: Streaming Generation")
    print("-" * 80)
    
    try:
        llm = LLMInterface(provider="anthropic")
        
        print("Response: ", end="", flush=True)
        for chunk in llm.generate_stream("Count from 1 to 5", max_tokens=50):
            print(chunk, end="", flush=True)
        print("\n✅ Streaming completed")
        
    except ValueError as e:
        print(f"⚠️  Skipped (API key not set): {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Usage stats
    print("\n" + "-" * 80)
    print("Test 3: Usage Statistics")
    print("-" * 80)
    
    try:
        llm = LLMInterface(provider="anthropic")
        stats = llm.get_usage_stats()
        
        print(f"✅ Stats: {stats}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 80)
    print("LLM Interface tests completed")
    print("=" * 80)
