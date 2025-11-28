"""
Configuration Management for NOVA AI System

This module provides centralized configuration management using Pydantic for:
- Environment variable loading and validation
- Type-safe configuration access
- Default values for all settings
- API key validation
- Model and performance parameters

The Settings class is a singleton that loads configuration from:
1. Environment variables (.env file)
2. Default values defined in this module

Usage:
    from src.config.settings import get_settings
    
    settings = get_settings()
    api_key = settings.ANTHROPIC_API_KEY
    model = settings.DEFAULT_LLM_MODEL

Author: NOVA Development Team
Date: 28 November 2025
"""

import os
from typing import Optional, Literal
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator, model_validator


class Settings(BaseSettings):
    """
    Main configuration class for NOVA AI System
    
    This class uses Pydantic BaseSettings to automatically load environment
    variables and provide type validation. All settings have sensible defaults
    that can be overridden via environment variables.
    
    Categories:
    - API Keys: Authentication for external services
    - Model Configuration: Default models and parameters
    - Vector Database: ChromaDB settings
    - Text Processing: Chunking and tokenization
    - Performance: Concurrency and optimization
    - Sampling: Text generation parameters
    - Logging: Application logging configuration
    - Development: Debug and device settings
    
    Attributes:
        See individual attribute docstrings below for detailed information.
    """
    
    # =========================================================================
    # API KEYS (Required for production)
    # =========================================================================
    
    ANTHROPIC_API_KEY: Optional[str] = Field(
        default=None,
        description="Anthropic Claude API key for text and vision models. "
                    "Get yours at: https://console.anthropic.com/"
    )
    
    MISTRAL_API_KEY: Optional[str] = Field(
        default=None,
        description="Mistral AI API key for LLM and embeddings. "
                    "Get yours at: https://console.mistral.ai/"
    )
    
    @field_validator('ANTHROPIC_API_KEY', 'MISTRAL_API_KEY')
    @classmethod
    def validate_api_key(cls, v: Optional[str]) -> Optional[str]:
        """
        Validate API key format
        
        Checks that API keys:
        - Are not empty strings
        - Have minimum length (basic sanity check)
        - Don't contain obvious placeholder text
        
        Args:
            v: The API key value
            
        Returns:
            Validated API key or None
            
        Raises:
            ValueError: If API key format is invalid
        """
        if v is None:
            return None
            
        # Strip whitespace
        v = v.strip()
        
        # Check for empty or placeholder values
        if not v or v in ['your-key-here', 'sk-ant-your-key-here', 'your-mistral-key-here']:
            return None
        
        # Minimum length check
        if len(v) < 10:
            raise ValueError(f"{field.name} appears to be too short. Expected valid API key.")
        
        return v
    
    # =========================================================================
    # MODEL CONFIGURATIONS
    # =========================================================================
    
    DEFAULT_LLM_MODEL: str = Field(
        default="claude-3-5-sonnet-20241022",
        description="Default LLM model for text generation. "
                    "Options: claude-3-5-sonnet-20241022, mistral-large-latest"
    )
    
    DEFAULT_EMBEDDING_MODEL: str = Field(
        default="mistral-embed",
        description="Default embedding model for semantic search. "
                    "Options: mistral-embed (1024D), nomic-embed (768D)"
    )
    
    DEFAULT_VISION_MODEL: str = Field(
        default="claude-3-5-sonnet-20241022",
        description="Default vision model for image analysis. "
                    "Options: claude-3-5-sonnet-20241022"
    )
    
    FALLBACK_LLM_MODEL: str = Field(
        default="mistral-small-latest",
        description="Fallback LLM model if primary fails or for cost optimization"
    )
    
    @field_validator('DEFAULT_LLM_MODEL', 'FALLBACK_LLM_MODEL')
    @classmethod
    def validate_llm_model(cls, v: str) -> str:
        """
        Validate LLM model name
        
        Ensures the model name is one of the supported models.
        
        Args:
            v: Model name
            
        Returns:
            Validated model name
            
        Raises:
            ValueError: If model name is not supported
        """
        supported_models = [
            'claude-3-5-sonnet-20241022',
            'claude-3-haiku-20240307',
            'mistral-large-latest',
            'mistral-small-latest',
        ]
        
        if v not in supported_models:
            raise ValueError(
                f"Unsupported LLM model: {v}. "
                f"Supported models: {', '.join(supported_models)}"
            )
        
        return v
    
    # =========================================================================
    # VECTOR DATABASE CONFIGURATION
    # =========================================================================
    
    CHROMA_PERSIST_DIRECTORY: str = Field(
        default="./data/chroma_db",
        description="Directory for persistent ChromaDB storage"
    )
    
    EMBEDDING_DIMENSION: int = Field(
        default=1024,
        ge=128,  # Greater than or equal to 128
        le=4096,  # Less than or equal to 4096
        description="Dimension of embedding vectors. Mistral: 1024, Nomic: 768"
    )
    
    COLLECTION_NAME_DOCUMENTS: str = Field(
        default="documents",
        description="ChromaDB collection name for document chunks"
    )
    
    COLLECTION_NAME_IMAGES: str = Field(
        default="images",
        description="ChromaDB collection name for image analyses"
    )
    
    COLLECTION_NAME_AUDIO: str = Field(
        default="audio",
        description="ChromaDB collection name for audio transcriptions"
    )
    
    @field_validator('CHROMA_PERSIST_DIRECTORY')
    @classmethod
    def validate_persist_directory(cls, v: str) -> str:
        """
        Validate and create persistence directory if it doesn't exist
        
        Args:
            v: Directory path
            
        Returns:
            Absolute path to directory
        """
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return str(path.absolute())
    
    # =========================================================================
    # TEXT PROCESSING CONFIGURATION
    # =========================================================================
    
    CHUNK_SIZE: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Target size for text chunks in characters. "
                    "Smaller = more granular, Larger = more context"
    )
    
    CHUNK_OVERLAP: int = Field(
        default=200,
        ge=0,
        le=1000,
        description="Overlap between consecutive chunks to preserve context"
    )
    
    CHUNK_STRATEGY: Literal["fixed", "paragraph", "semantic"] = Field(
        default="paragraph",
        description="Chunking strategy: fixed (char count), paragraph (natural), semantic (embedding-based)"
    )
    
    MAX_TOKENS_PER_CHUNK: int = Field(
        default=512,
        ge=64,
        le=8192,
        description="Maximum tokens per chunk (for model context limits)"
    )
    
    @model_validator(mode='after')
    def validate_overlap(self) -> 'Settings':
        """
        Ensure overlap is less than chunk size
        
        Returns:
            Self after validation
            
        Raises:
            ValueError: If overlap >= chunk_size
        """
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.CHUNK_OVERLAP}) must be less than CHUNK_SIZE ({self.CHUNK_SIZE})"
            )
        return self
    
    # =========================================================================
    # PERFORMANCE SETTINGS
    # =========================================================================
    
    MAX_CONCURRENT_REQUESTS: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of concurrent API requests"
    )
    
    REQUEST_TIMEOUT: int = Field(
        default=60,
        ge=10,
        le=300,
        description="API request timeout in seconds"
    )
    
    BATCH_SIZE: int = Field(
        default=8,
        ge=1,
        le=128,
        description="Batch size for embedding generation"
    )
    
    USE_KV_CACHE: bool = Field(
        default=True,
        description="Enable KV cache for autoregressive generation (10-100x speedup)"
    )
    
    ENABLE_STREAMING: bool = Field(
        default=True,
        description="Enable streaming responses for better UX"
    )
    
    # =========================================================================
    # SAMPLING PARAMETERS (Text Generation)
    # =========================================================================
    
    DEFAULT_TEMPERATURE: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. Lower = more focused, Higher = more random"
    )
    
    DEFAULT_TOP_K: Optional[int] = Field(
        default=50,
        ge=1,
        le=500,
        description="Top-K sampling: keep only K most probable tokens. None = disabled"
    )
    
    DEFAULT_TOP_P: Optional[float] = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Top-P (nucleus) sampling: cumulative probability threshold. None = disabled"
    )
    
    DEFAULT_MAX_TOKENS: int = Field(
        default=2000,
        ge=1,
        le=100000,
        description="Maximum tokens to generate in response"
    )
    
    REPETITION_PENALTY: float = Field(
        default=1.1,
        ge=1.0,
        le=2.0,
        description="Repetition penalty. 1.0 = no penalty, higher = less repetition"
    )
    
    @model_validator(mode='after')
    def validate_sampling_params(self) -> 'Settings':
        """
        Validate that sampling parameters are coherent
        
        Ensures:
        - At least one of top-k or top-p is enabled
        - Temperature is reasonable for the task
        
        Returns:
            Self after validation
        """
        # Warn if both are disabled (will use pure temperature sampling)
        if self.DEFAULT_TOP_K is None and self.DEFAULT_TOP_P is None:
            import warnings
            warnings.warn(
                "Both TOP_K and TOP_P are disabled. Using pure temperature sampling. "
                "This may produce lower quality outputs."
            )
        
        return self
    
    # =========================================================================
    # LOGGING CONFIGURATION
    # =========================================================================
    
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level"
    )
    
    LOG_FILE: str = Field(
        default="./logs/nova.log",
        description="Path to log file"
    )
    
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )
    
    LOG_DATE_FORMAT: str = Field(
        default="%Y-%m-%d %H:%M:%S",
        description="Log timestamp format"
    )
    
    VERBOSE_LOGGING: bool = Field(
        default=False,
        description="Enable verbose logging (includes API request/response details)"
    )
    
    @field_validator('LOG_FILE')
    @classmethod
    def validate_log_file(cls, v: str) -> str:
        """
        Create log directory if it doesn't exist
        
        Args:
            v: Log file path
            
        Returns:
            Absolute path to log file
        """
        log_path = Path(v)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        return str(log_path.absolute())
    
    # =========================================================================
    # DEVELOPMENT & DEBUGGING
    # =========================================================================
    
    DEBUG: bool = Field(
        default=False,
        description="Enable debug mode (more verbose output, no API calls in dry-run)"
    )
    
    DEVICE: Literal["cpu", "cuda", "mps"] = Field(
        default="cpu",
        description="PyTorch device for local model inference. "
                    "Options: cpu (all platforms), cuda (NVIDIA), mps (Apple Silicon)"
    )
    
    SEED: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility. None = random"
    )
    
    PROFILE_PERFORMANCE: bool = Field(
        default=False,
        description="Enable performance profiling (measures execution time)"
    )
    
    @field_validator('DEVICE')
    @classmethod
    def validate_device(cls, v: str) -> str:
        """
        Validate that requested device is available
        
        Args:
            v: Device name
            
        Returns:
            Validated device name
            
        Raises:
            ValueError: If device is not available
        """
        import torch
        
        if v == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                "CUDA device requested but not available. "
                "Install CUDA or use 'cpu' device."
            )
        
        if v == "mps" and not torch.backends.mps.is_available():
            raise ValueError(
                "MPS (Metal Performance Shaders) device requested but not available. "
                "Use 'cpu' device or check macOS version (requires macOS 12.3+)."
            )
        
        return v
    
    # =========================================================================
    # PYDANTIC CONFIGURATION
    # =========================================================================
    
    class Config:
        """
        Pydantic configuration class
        
        Defines how settings are loaded and validated:
        - env_file: Load from .env file
        - env_file_encoding: UTF-8 encoding
        - case_sensitive: Environment variable names are case-sensitive
        """
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        
        # Allow arbitrary types (for Path, etc.)
        arbitrary_types_allowed = True
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_model_config(self, model_type: str = "llm") -> dict:
        """
        Get model configuration as dictionary
        
        Useful for passing to API clients or model initialization.
        
        Args:
            model_type: Type of model ("llm", "embedding", "vision")
            
        Returns:
            Dictionary with model configuration
            
        Example:
            >>> settings = get_settings()
            >>> llm_config = settings.get_model_config("llm")
            >>> print(llm_config)
            {
                'model': 'claude-3-5-sonnet-20241022',
                'max_tokens': 2000,
                'temperature': 0.7,
                'top_k': 50,
                'top_p': 0.9
            }
        """
        if model_type == "llm":
            return {
                'model': self.DEFAULT_LLM_MODEL,
                'max_tokens': self.DEFAULT_MAX_TOKENS,
                'temperature': self.DEFAULT_TEMPERATURE,
                'top_k': self.DEFAULT_TOP_K,
                'top_p': self.DEFAULT_TOP_P,
            }
        elif model_type == "embedding":
            return {
                'model': self.DEFAULT_EMBEDDING_MODEL,
                'dimension': self.EMBEDDING_DIMENSION,
            }
        elif model_type == "vision":
            return {
                'model': self.DEFAULT_VISION_MODEL,
                'max_tokens': self.DEFAULT_MAX_TOKENS,
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def get_chunking_config(self) -> dict:
        """
        Get text chunking configuration
        
        Returns:
            Dictionary with chunking parameters
            
        Example:
            >>> settings = get_settings()
            >>> chunking = settings.get_chunking_config()
            >>> print(chunking)
            {
                'chunk_size': 1000,
                'chunk_overlap': 200,
                'strategy': 'paragraph',
                'max_tokens': 512
            }
        """
        return {
            'chunk_size': self.CHUNK_SIZE,
            'chunk_overlap': self.CHUNK_OVERLAP,
            'strategy': self.CHUNK_STRATEGY,
            'max_tokens': self.MAX_TOKENS_PER_CHUNK,
        }
    
    def is_production_ready(self) -> tuple[bool, list[str]]:
        """
        Check if configuration is ready for production
        
        Validates that all required API keys are set and configuration
        is sensible for production use.
        
        Returns:
            Tuple of (is_ready: bool, issues: list[str])
            
        Example:
            >>> settings = get_settings()
            >>> ready, issues = settings.is_production_ready()
            >>> if not ready:
            ...     print("Configuration issues:", issues)
        """
        issues = []
        
        # Check API keys
        if not self.ANTHROPIC_API_KEY:
            issues.append("ANTHROPIC_API_KEY is not set")
        
        if not self.MISTRAL_API_KEY:
            issues.append("MISTRAL_API_KEY is not set")
        
        # Check debug mode
        if self.DEBUG:
            issues.append("DEBUG mode is enabled (should be False in production)")
        
        # Check verbose logging
        if self.VERBOSE_LOGGING:
            issues.append("VERBOSE_LOGGING is enabled (may leak sensitive data)")
        
        # Check reasonable timeouts
        if self.REQUEST_TIMEOUT < 30:
            issues.append(f"REQUEST_TIMEOUT is very short ({self.REQUEST_TIMEOUT}s)")
        
        return len(issues) == 0, issues
    
    def __repr__(self) -> str:
        """
        String representation (hides API keys for security)
        
        Returns:
            Safe string representation
        """
        return (
            f"Settings("
            f"llm_model={self.DEFAULT_LLM_MODEL}, "
            f"embedding_model={self.DEFAULT_EMBEDDING_MODEL}, "
            f"device={self.DEVICE}, "
            f"debug={self.DEBUG})"
        )


# =============================================================================
# SINGLETON PATTERN
# =============================================================================

@lru_cache()
def get_settings() -> Settings:
    """
    Get singleton Settings instance
    
    Uses LRU cache to ensure only one Settings instance is created
    and reused throughout the application lifecycle.
    
    This pattern ensures:
    1. Configuration is loaded only once
    2. All modules use the same configuration
    3. Changes to .env require application restart
    
    Returns:
        Singleton Settings instance
        
    Example:
        >>> from src.config.settings import get_settings
        >>> settings = get_settings()
        >>> print(settings.DEFAULT_LLM_MODEL)
        claude-3-5-sonnet-20241022
    """
    return Settings()


# =============================================================================
# MODULE-LEVEL CONVENIENCE
# =============================================================================

# For direct import: from src.config.settings import settings
settings = get_settings()


if __name__ == "__main__":
    """
    Test configuration loading
    
    Run this module directly to validate configuration:
        python -m src.config.settings
    """
    import sys
    
    print("=" * 80)
    print("NOVA AI System - Configuration Test")
    print("=" * 80)
    
    try:
        config = get_settings()
        print(f"\n✅ Configuration loaded successfully!\n")
        print(f"Representation: {config}\n")
        
        # Check production readiness
        ready, issues = config.is_production_ready()
        
        if ready:
            print("✅ Configuration is production-ready!\n")
        else:
            print("⚠️  Configuration issues found:\n")
            for issue in issues:
                print(f"  - {issue}")
            print()
        
        # Display key settings
        print("Key Settings:")
        print(f"  LLM Model: {config.DEFAULT_LLM_MODEL}")
        print(f"  Embedding Model: {config.DEFAULT_EMBEDDING_MODEL}")
        print(f"  Embedding Dimension: {config.EMBEDDING_DIMENSION}")
        print(f"  Device: {config.DEVICE}")
        print(f"  KV Cache: {'Enabled' if config.USE_KV_CACHE else 'Disabled'}")
        print(f"  Log Level: {config.LOG_LEVEL}")
        print(f"  Debug Mode: {'Enabled' if config.DEBUG else 'Disabled'}")
        
        print("\nModel Configuration:")
        print(f"  LLM: {config.get_model_config('llm')}")
        print(f"  Embedding: {config.get_model_config('embedding')}")
        
        print("\nChunking Configuration:")
        print(f"  {config.get_chunking_config()}")
        
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Configuration error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
