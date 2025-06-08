"""Enhanced application settings optimized for cost-effective OpenAI usage."""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from enum import Enum


class LogLevel(str, Enum):
    """Supported logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class SupportedModel(str, Enum):
    """Supported LLM models optimized for cost."""
    # Most cost-effective models (cheapest first)
    GPT_4_1_NANO = "gpt-4.1-nano-2025-04-14"           # $0.10/$0.40 - Best value
    GPT_4O_MINI = "gpt-4o-mini-2024-07-18"             # $0.15/$0.60 - Good backup
    GPT_4_1_MINI = "gpt-4.1-mini-2025-04-14"           # $0.40/$1.60 - If nano insufficient
    
    # Fallback options (more expensive)
    GPT_4_1 = "gpt-4.1-2025-04-14"                     # $2.00/$8.00
    GPT_4O = "gpt-4o-2024-08-06"                       # $2.50/$10.00
    
    # Legacy (for compatibility)
    GPT_3_5_TURBO = "gpt-3.5-turbo"                    # Being deprecated


class ExportFormat(str, Enum):
    """Supported export formats."""
    JSON = "json"
    CSV = "csv"
    XLSX = "xlsx"
    XML = "xml"


class Settings(BaseSettings):
    """Application settings optimized for cost-effective OpenAI usage."""
    
    # Project Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    
    # OpenAI Configuration - Cost Optimized
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_model: SupportedModel = Field(
        SupportedModel.GPT_4_1_NANO, 
        description="Primary model - optimized for cost"
    )
    openai_fallback_model: SupportedModel = Field(
        SupportedModel.GPT_4O_MINI,
        description="Fallback model if primary fails"
    )
    
    # Token Configuration - Optimized for cost
    openai_max_tokens: int = Field(
        1500, 
        ge=100, 
        le=4000,
        description="Lower max tokens to reduce costs"
    )
    openai_temperature: float = Field(
        0.0, 
        ge=0.0, 
        le=1.0,
        description="Lower temperature for more consistent, focused responses"
    )
    
    # Cost Control Settings
    enable_caching: bool = Field(
        True, 
        description="Enable caching to leverage cheaper cached input pricing"
    )
    cache_ttl_hours: int = Field(
        72, 
        ge=1,
        description="Longer cache for cost savings"
    )
    max_daily_tokens: int = Field(
        100000, 
        description="Daily token limit for cost control"
    )
    token_budget_alert_threshold: float = Field(
        0.8, 
        ge=0.1, 
        le=1.0,
        description="Alert when reaching this % of daily budget"
    )
    
    # Alternative LLM Providers (for cost comparison)
    anthropic_api_key: Optional[str] = Field(default=None)
    azure_openai_endpoint: Optional[str] = Field(default=None)
    azure_openai_api_key: Optional[str] = Field(default=None)
    
    # Database Configuration
    database_url: str = Field(default="sqlite:///data/database/bioeconomic_products.db")
    database_echo: bool = Field(default=False)
    
    # Logging Configuration
    log_level: LogLevel = Field(default=LogLevel.INFO)
    log_file: str = Field(default="logs/bioeconomic_analyzer.log")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    log_max_bytes: int = Field(default=10485760)  # 10MB
    log_backup_count: int = Field(default=5)
    
    # Processing Configuration - Optimized for cost
    max_chunk_size: int = Field(
        default=2500, 
        ge=500, 
        le=8000,
        description="Smaller chunks to fit in nano model context efficiently"
    )
    chunk_overlap: int = Field(
        default=150, 
        ge=0,
        description="Reduced overlap to minimize token usage"
    )
    batch_size: int = Field(
        default=5, 
        ge=1, 
        le=20,
        description="Smaller batches for better cost control"
    )
    max_retries: int = Field(default=2, ge=0, le=5, description="Fewer retries to reduce costs")
    retry_delay: float = Field(default=2.0, ge=0.5, description="Longer delay to avoid rate limits")
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    
    # Performance Configuration - Cost optimized
    max_concurrent_requests: int = Field(
        default=2, 
        ge=1, 
        le=10,
        description="Lower concurrency to avoid rate limits and manage costs"
    )
    request_timeout_seconds: int = Field(default=45, ge=10)
    
    # Smart Processing - Cost optimization features
    enable_smart_chunking: bool = Field(
        default=True, 
        description="Use intelligent chunking to minimize token usage"
    )
    enable_result_deduplication: bool = Field(
        default=True, 
        description="Avoid processing duplicate content"
    )
    enable_progressive_extraction: bool = Field(
        default=True, 
        description="Extract incrementally to optimize token usage"
    )
    
    # File Processing - Fix the problematic field
    supported_formats: str = Field(
        default="pdf,txt,docx",
        description="Comma-separated list of supported file formats"
    )
    max_file_size_mb: int = Field(default=25, ge=1, le=100, description="Smaller max size for cost control")
    temp_dir: str = Field(default="data/temp")
    
    # Export Configuration
    default_export_format: ExportFormat = Field(default=ExportFormat.JSON)
    include_metadata: bool = Field(default=True)
    pretty_print_json: bool = Field(default=False, description="Compact JSON to save space")
    
    # Development/Debug Settings
    debug_mode: bool = Field(default=False)
    verbose_logging: bool = Field(default=False)
    save_raw_responses: bool = Field(default=False)
    enable_profiling: bool = Field(default=False)
    
    # Cost monitoring
    enable_cost_tracking: bool = Field(default=True)
    cost_alert_threshold_usd: float = Field(default=5.0, ge=0.1)
    
    @field_validator("supported_formats", mode="after")
    @classmethod
    def parse_supported_formats(cls, v):
        """Parse supported formats from string."""
        if isinstance(v, str):
            return [fmt.strip().lower() for fmt in v.split(",") if fmt.strip()]
        elif isinstance(v, list):
            return [fmt.lower() for fmt in v if fmt]
        return ["pdf", "txt", "docx"]  # Default fallback
    
    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, v, info):
        """Ensure chunk overlap is smaller than chunk size."""
        # Access other field values properly in Pydantic v2
        if hasattr(info, 'data') and 'max_chunk_size' in info.data:
            max_chunk_size = info.data['max_chunk_size']
            if v >= max_chunk_size:
                raise ValueError("chunk_overlap must be smaller than max_chunk_size")
        return v
    
    @field_validator("log_file", "temp_dir", mode="before")
    @classmethod
    def resolve_paths(cls, v):
        """Resolve relative paths."""
        if isinstance(v, str) and not os.path.isabs(v):
            return os.path.join(Path(__file__).parent.parent, v)
        return v
    
    @property
    def data_dir(self) -> Path:
        """Get data directory path."""
        return self.project_root / "data"
    
    @property
    def input_dir(self) -> Path:
        """Get input directory path."""
        return self.data_dir / "input" / "pdfs"
    
    @property
    def output_dir(self) -> Path:
        """Get output directory path."""
        return self.data_dir / "output"
    
    @property
    def database_dir(self) -> Path:
        """Get database directory path."""
        return self.data_dir / "database"
    
    @property
    def logs_dir(self) -> Path:
        """Get logs directory path."""
        return self.project_root / "logs"
    
    @property
    def exports_dir(self) -> Path:
        """Get exports directory path."""
        return self.data_dir / "exports"
    
    @property
    def cache_dir(self) -> Path:
        """Get cache directory path."""
        return self.data_dir / "cache"
    
    @property 
    def supported_formats_list(self) -> List[str]:
        """Get supported formats as a list."""
        if isinstance(self.supported_formats, str):
            return [fmt.strip().lower() for fmt in self.supported_formats.split(",") if fmt.strip()]
        return self.supported_formats
    
    def create_directories(self):
        """Create all necessary directories."""
        directories = [
            self.data_dir,
            self.input_dir,
            self.output_dir,
            self.database_dir,
            self.logs_dir,
            self.exports_dir,
            self.cache_dir,
            Path(self.temp_dir)
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_model_pricing(self) -> Dict[str, Dict[str, float]]:
        """Get pricing information for supported models (per 1M tokens)."""
        return {
            "gpt-4.1-nano-2025-04-14": {
                "input": 0.10,
                "cached_input": 0.025,
                "output": 0.40
            },
            "gpt-4o-mini-2024-07-18": {
                "input": 0.15,
                "cached_input": 0.075,
                "output": 0.60
            },
            "gpt-4.1-mini-2025-04-14": {
                "input": 0.40,
                "cached_input": 0.10,
                "output": 1.60
            },
            "gpt-4.1-2025-04-14": {
                "input": 2.00,
                "cached_input": 0.50,
                "output": 8.00
            },
            "gpt-4o-2024-08-06": {
                "input": 2.50,
                "cached_input": 1.25,
                "output": 10.00
            }
        }
    
    def estimate_cost(self, input_tokens: int, output_tokens: int, model: Optional[str] = None, use_cache: bool = False) -> float:
        """
        Estimate cost for a request.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model to use (defaults to current model)
            use_cache: Whether cached input pricing applies
            
        Returns:
            Estimated cost in USD
        """
        model = model or self.openai_model.value
        pricing = self.get_model_pricing().get(model, {})
        
        if not pricing:
            return 0.0
        
        input_price = pricing.get("cached_input" if use_cache else "input", 0.0)
        output_price = pricing.get("output", 0.0)
        
        cost = (input_tokens * input_price / 1_000_000) + (output_tokens * output_price / 1_000_000)
        return round(cost, 6)
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration dictionary."""
        return {
            "url": self.database_url,
            "echo": self.database_echo,
            "pool_pre_ping": True,
            "pool_recycle": 300
        }
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration dictionary."""
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": self.log_format
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "level": self.log_level.value,
                    "class": "logging.StreamHandler",
                    "formatter": "standard"
                },
                "file": {
                    "level": self.log_level.value,
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": self.log_file,
                    "maxBytes": self.log_max_bytes,
                    "backupCount": self.log_backup_count,
                    "formatter": "detailed"
                }
            },
            "loggers": {
                "": {
                    "handlers": ["console", "file"],
                    "level": self.log_level.value,
                    "propagate": False
                }
            }
        }
    
    model_config = {
        "env_file": Path(__file__).parent.parent / ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
        "validate_assignment": True
    }


# Configuration loader with validation
class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self):
        self._settings = None
    
    @property
    def settings(self) -> Settings:
        """Get settings instance (lazy loaded)."""
        if self._settings is None:
            self._settings = self._load_settings()
        return self._settings
    
    def _load_settings(self) -> Settings:
        """Load and validate settings."""
        try:
            settings = Settings()
            settings.create_directories()
            return settings
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {e}")
    
    def reload_settings(self) -> Settings:
        """Reload settings (useful for tests or config changes)."""
        self._settings = None
        return self.settings
    
    def validate_api_keys(self) -> Dict[str, bool]:
        """Validate that required API keys are present."""
        validation = {}
        
        # OpenAI is required
        validation["openai"] = bool(self.settings.openai_api_key and 
                                  self.settings.openai_api_key != "your_openai_api_key_here")
        
        # Other providers are optional
        validation["anthropic"] = bool(self.settings.anthropic_api_key)
        validation["azure"] = bool(self.settings.azure_openai_endpoint and 
                                 self.settings.azure_openai_api_key)
        
        return validation
    
    def get_active_providers(self) -> List[str]:
        """Get list of configured LLM providers."""
        providers = []
        validation = self.validate_api_keys()
        
        if validation["openai"]:
            providers.append("openai")
        if validation["anthropic"]:
            providers.append("anthropic")
        if validation["azure"]:
            providers.append("azure")
        
        return providers
    
    def get_cost_optimization_tips(self) -> List[str]:
        """Get cost optimization recommendations."""
        tips = []
        settings = self.settings
        
        if settings.openai_model.value != SupportedModel.GPT_4_1_NANO.value:
            tips.append(f"💡 Consider using {SupportedModel.GPT_4_1_NANO.value} for lowest costs ($0.10/$0.40 per 1M tokens)")
        
        if settings.openai_max_tokens > 2000:
            tips.append("💡 Reduce max_tokens to lower output costs")
            
        if not settings.enable_caching:
            tips.append("💡 Enable caching to leverage 75% cheaper cached input pricing")
            
        if settings.openai_temperature > 0.2:
            tips.append("💡 Lower temperature for more focused, shorter responses")
            
        if settings.max_concurrent_requests > 3:
            tips.append("💡 Reduce concurrent requests to avoid rate limits and better cost control")
            
        return tips


# Global configuration manager instance
config_manager = ConfigManager()

# Convenience access to settings
settings = config_manager.settings