"""
Bioeconomic Products Analysis System

A comprehensive system for extracting, analyzing, and managing information about 
bioeconomic products from Latin America and the Caribbean region.

This package provides:
- Document processing and text extraction
- LLM-based product information extraction 
- Data storage and management
- Export capabilities in multiple formats
- Comprehensive analysis and reporting tools

Main Components:
- extractors: Product extraction implementations
- llm: Language model interaction modules
- models: Data models and schemas
- storage: Database and export management
- utils: Utility functions and text preprocessing
"""

import logging
from pathlib import Path

# Package metadata
__version__ = "1.0.0"
__author__ = "Bioeconomic Products Analysis Team"
__description__ = "System for analyzing bioeconomic products from Latin America and Caribbean"

# Main imports for easy access
from .models import (
    Product,
    ProductOrigin,
    ProductUse,
    ProductCategory,
    ProcessingLevel,
    ExtractionResult,
    ExtractionMetadata,
    ProcessingStatus
)

from .extractors import (
    BaseExtractor,
    ProductExtractor,
    ExtractorResult
)

from .storage import (
    DatabaseManager,
    ProductDatabase,
    ExportManager,
    ExportFormat
)

from .llm import (
    OpenAIClient,
    PanAmazonPromptManager as PromptManager,
    PanAmazonResponseParser as ResponseParser
)

from .utils import (
    setup_logging,
    TextPreprocessor,
    LanguageDetector,
    ContentFilter
)

# Configure package-level logging
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "database_path": Path("data/products.db"),
    "exports_path": Path("exports"),
    "logs_path": Path("logs"),
    "temp_path": Path("temp"),
    "chunk_size": 1000,
    "chunk_overlap": 100,
    "confidence_threshold": 0.5,
    "max_retries": 3,
    "log_level": "INFO"
}


def initialize_package(config: dict = None) -> dict:
    """
    Initialize the package with configuration.

    Args:
        config: Configuration dictionary (optional)

    Returns:
        Applied configuration
    """
    # Merge with defaults
    applied_config = DEFAULT_CONFIG.copy()
    if config:
        applied_config.update(config)

    # Create necessary directories
    for path_key in ["exports_path", "logs_path", "temp_path"]:
        if path_key in applied_config:
            path = applied_config[path_key]
            if isinstance(path, (str, Path)):
                Path(path).mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = applied_config.get("logs_path")
    if log_file:
        log_file = Path(log_file) / "bioeconomic_analysis.log"

    setup_logging(
        log_level=applied_config.get("log_level", "INFO"),
        log_file=log_file
    )

    logger.info(f"Bioeconomic Products Analysis System v{__version__} initialized")
    logger.debug(f"Configuration: {applied_config}")

    return applied_config


def get_version() -> str:
    """Get package version."""
    return __version__


def get_system_info() -> dict:
    """
    Get system information.

    Returns:
        Dictionary with system and package information
    """
    from .utils.helpers import get_system_info

    info = get_system_info()
    info.update({
        "package_version": __version__,
        "package_name": "bioeconomic_products_analysis"
    })

    return info


class BioeconomicAnalyzer:
    """
    Main analyzer class that coordinates all components.

    This is a high-level interface that combines extraction, storage,
    and analysis capabilities in a single, easy-to-use class.
    """

    def __init__(self, config: dict = None):
        """
        Initialize the analyzer.

        Args:
            config: Configuration dictionary
        """
        self.config = initialize_package(config)
        self.logger = logging.getLogger(f"{__name__}.BioeconomicAnalyzer")

        # Initialize components
        self._init_components()

    def _init_components(self):
        """Initialize all analyzer components."""
        try:
            # Database
            self.db = ProductDatabase()

            # Text preprocessor
            self.preprocessor = TextPreprocessor(
                chunk_size=self.config.get("chunk_size", 1000),
                overlap=self.config.get("chunk_overlap", 100)
            )

            # LLM components (if API key available)
            try:
                self.llm_client = OpenAIClient()
                self.prompt_manager = PromptManager()
                self.response_parser = ResponseParser()
                self.extractor = ProductExtractor()
                # Initialize with available components
                # self.extractor = ProductExtractor(
                #     client=self.llm_client,
                #     prompt_manager=self.prompt_manager,
                #     parser=self.response_parser
                # )
                self.llm_available = True
            except Exception as e:
                self.logger.warning(f"LLM components not available: {e}")
                self.llm_available = False
                self.extractor = None

            # Export manager
            self.export_manager = ExportManager(self.db)

            self.logger.info("Analyzer components initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise

    def extract_from_text(self, text: str, **kwargs) -> 'ExtractionResult':
        """
        Extract products from text.

        Args:
            text: Text to analyze
            **kwargs: Additional extraction parameters

        Returns:
            ExtractionResult with extracted products
        """
        if not self.llm_available or not self.extractor:
            raise RuntimeError("LLM extractor not available")

        self.logger.info(f"Starting extraction from text ({len(text)} characters)")

        try:
            # Preprocess text
            processed = self.preprocessor.process(text)

            # Extract products
            extraction_result = self.extractor.extract(
                processed['processed_text'],
                **kwargs
            )

            # Save to database
            if extraction_result.products:
                self.db.save_extraction_result(extraction_result)
                self.logger.info(f"Saved {len(extraction_result.products)} products to database")

            return extraction_result

        except Exception as e:
            self.logger.error(f"Extraction failed: {e}")
            raise

    def extract_from_file(self, file_path: Path, **kwargs) -> 'ExtractionResult':
        """
        Extract products from file.

        Args:
            file_path: Path to file to analyze
            **kwargs: Additional extraction parameters

        Returns:
            ExtractionResult with extracted products
        """
        self.logger.info(f"Starting extraction from file: {file_path}")

        try:
            # Read file content (this would need file processing logic)
            # For now, assume it's a text file
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # Add file metadata
            kwargs['source_file'] = str(file_path)

            return self.extract_from_text(text, **kwargs)

        except Exception as e:
            self.logger.error(f"File extraction failed: {e}")
            raise

    def search_products(self, **filters) -> list:
        """
        Search products in database.

        Args:
            **filters: Search filters (name_query, country, category, etc.)

        Returns:
            List of matching products
        """
        return self.db.search_products(**filters)

    def export_products(self, products: list, output_path: Path, format: ExportFormat) -> bool:
        """
        Export products to file.

        Args:
            products: List of products to export
            output_path: Output file path
            format: Export format

        Returns:
            True if export successful
        """
        return self.export_manager.export_products(products, output_path, format)

    def get_statistics(self) -> dict:
        """
        Get analysis statistics.

        Returns:
            Dictionary with statistics
        """
        db_stats = self.db.get_statistics()
        system_stats = get_system_info()

        return {
            "database": db_stats,
            "system": system_stats,
            "analyzer": {
                "llm_available": self.llm_available,
                "version": __version__
            }
        }


# Export main classes for easy import
__all__ = [
    # Core classes
    "BioeconomicAnalyzer",

    # Models
    "Product",
    "ProductOrigin", 
    "ProductUse",
    "ProductCategory",
    "ProcessingLevel",
    "ExtractionResult",
    "ExtractionMetadata",
    "ProcessingStatus",

    # Extractors
    "BaseExtractor",
    "ProductExtractor",
    "ExtractorResult",

    # Storage
    "DatabaseManager",
    "ProductDatabase",
    "ExportManager", 
    "ExportFormat",

    # LLM
    "OpenAIClient",
    "PromptManager",
    "ResponseParser",

    # Utils
    "setup_logging",
    "TextPreprocessor",
    "LanguageDetector",
    "ContentFilter",

    # Functions
    "initialize_package",
    "get_version",
    "get_system_info",

    # Constants
    "__version__",
    "__author__",
    "__description__"
]


# Initialize package when imported
_default_config = initialize_package()
