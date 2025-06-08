"""Product extractor for bioeconomic products analysis."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

# Use absolute imports to avoid import issues
from config.settings import settings

# Use relative imports for prompt manager
from ..llm.prompt_manager import PanAmazonPromptManager

logger = logging.getLogger(__name__)

@dataclass
class ExtractorResult:
    """Result container for product extraction."""
    products: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    confidence: float
    processing_time: float

class ProductExtractor:
    """Extractor for bioeconomic products from text documents."""

    def __init__(self):
        """Initialize the product extractor."""
        self.settings = settings
        # Create prompt manager instance directly
        self.prompt_manager = PanAmazonPromptManager()

    def extract_products(self, text: str, source_name: str = "documento") -> ExtractorResult:
        """
        Extract products from the given text.

        Args:
            text: The input text to analyze
            source_name: Name of the source document

        Returns:
            ExtractorResult containing extracted products and metadata
        """
        import time
        start_time = time.time()

        try:
            # Use the prompt manager to format extraction prompt
            formatted_prompt = self.prompt_manager.format_extraction_prompt(text, source_name)

            # Here you would implement the actual extraction logic
            # This is a placeholder implementation
            products = []
            metadata = {
                "source": source_name,
                "text_length": len(text),
                "prompt_tokens": len(formatted_prompt["system"] + formatted_prompt["user"]) // 4
            }

            processing_time = time.time() - start_time

            return ExtractorResult(
                products=products,
                metadata=metadata,
                confidence=0.0,
                processing_time=processing_time
            )

        except Exception as e:
            logger.error(f"Error extracting products: {e}")
            processing_time = time.time() - start_time
            return ExtractorResult(
                products=[],
                metadata={"error": str(e)},
                confidence=0.0,
                processing_time=processing_time
            )
