"""Base extractor class for all extraction implementations."""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExtractorResult:
    """Container for extraction results."""
    products: List[Dict[str, Any]]
    confidence_score: float
    extraction_method: str
    tokens_used: Optional[int] = None
    processing_time: Optional[float] = None
    source_text_length: int = 0


class BaseExtractor(ABC):
    """Abstract base class for all extractors."""

    def __init__(self, name: str):
        """
        Initialize the base extractor.

        Args:
            name: Name identifier for this extractor
        """
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")

    @abstractmethod
    def extract(self, text: str, **kwargs) -> ExtractorResult:
        """
        Extract products from text.

        Args:
            text: Text to extract products from
            **kwargs: Additional parameters specific to implementation

        Returns:
            ExtractorResult with extracted products and metadata
        """
        pass

    def validate_json_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Validate and parse JSON response from LLM.

        Args:
            response: Raw response string from LLM

        Returns:
            List of product dictionaries

        Raises:
            ValueError: If response is not valid JSON or doesn't match expected format
        """
        try:
            # Clean response - remove any text before/after JSON
            response = response.strip()

            # Find JSON array boundaries
            start_idx = response.find('[')
            end_idx = response.rfind(']')

            if start_idx == -1 or end_idx == -1:
                raise ValueError("No JSON array found in response")

            json_str = response[start_idx:end_idx + 1]
            products = json.loads(json_str)

            if not isinstance(products, list):
                raise ValueError("Response must be a JSON array")

            # Validate each product has required fields
            validated_products = []
            for product in products:
                validated_product = self._validate_product_schema(product)
                if validated_product:
                    validated_products.append(validated_product)

            return validated_products

        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in response: {e}")
            raise ValueError(f"Invalid JSON response: {e}")
        except Exception as e:
            self.logger.error(f"Error validating response: {e}")
            raise ValueError(f"Response validation failed: {e}")

    def _validate_product_schema(self, product: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Validate individual product schema.

        Args:
            product: Product dictionary to validate

        Returns:
            Validated product dictionary or None if invalid
        """
        required_fields = ['product_name', 'country', 'uses']
        optional_fields = ['scientific_name', 'additional_info']

        if not isinstance(product, dict):
            self.logger.warning("Product is not a dictionary")
            return None

        # Check required fields
        for field in required_fields:
            if field not in product or not product[field]:
                self.logger.warning(f"Product missing required field: {field}")
                return None

        # Clean and standardize the product
        validated = {}

        # Required fields
        validated['product_name'] = str(product['product_name']).strip()
        validated['country'] = str(product['country']).strip()

        # Handle uses field (can be string or list)
        uses = product['uses']
        if isinstance(uses, str):
            validated['uses'] = [uses.strip()]
        elif isinstance(uses, list):
            validated['uses'] = [str(use).strip() for use in uses if str(use).strip()]
        else:
            validated['uses'] = [str(uses).strip()]

        # Optional fields
        for field in optional_fields:
            if field in product and product[field]:
                validated[field] = str(product[field]).strip()
            else:
                validated[field] = ""

        # Basic validation checks
        if len(validated['product_name']) < 2:
            self.logger.warning("Product name too short")
            return None

        if len(validated['country']) < 2:
            self.logger.warning("Country name too short")
            return None

        if not validated['uses'] or all(len(use) < 2 for use in validated['uses']):
            self.logger.warning("No valid uses found")
            return None

        return validated

    def calculate_confidence_score(self, products: List[Dict[str, Any]], text: str) -> float:
        """
        Calculate confidence score for extracted products.

        Args:
            products: List of extracted products
            text: Source text

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not products:
            return 0.0

        score = 0.5  # Base score for finding any products

        # Bonus for completeness
        complete_products = 0
        for product in products:
            completeness = 0
            if product.get('scientific_name'):
                completeness += 0.25
            if product.get('additional_info'):
                completeness += 0.25
            if len(product.get('uses', [])) > 1:
                completeness += 0.25
            if len(product.get('product_name', '')) > 5:
                completeness += 0.25

            if completeness > 0.5:
                complete_products += 1

        if complete_products > 0:
            score += (complete_products / len(products)) * 0.3

        # Bonus for finding product names in text
        found_in_text = 0
        text_lower = text.lower()
        for product in products:
            product_name = product.get('product_name', '').lower()
            if product_name and product_name in text_lower:
                found_in_text += 1

        if found_in_text > 0:
            score += (found_in_text / len(products)) * 0.2

        return min(1.0, score)

    def log_extraction_result(self, result: ExtractorResult, text_preview: str = ""):
        """
        Log extraction results for debugging.

        Args:
            result: Extraction result to log
            text_preview: Preview of source text (optional)
        """
        self.logger.info(f"Extraction completed: {len(result.products)} products found")
        self.logger.info(f"Confidence score: {result.confidence_score:.3f}")
        self.logger.info(f"Source text length: {result.source_text_length} characters")

        if result.tokens_used:
            self.logger.info(f"Tokens used: {result.tokens_used}")

        if result.processing_time:
            self.logger.info(f"Processing time: {result.processing_time:.2f}s")

        if result.products:
            self.logger.debug("Extracted products:")
            for i, product in enumerate(result.products, 1):
                self.logger.debug(f"  {i}. {product.get('product_name', 'Unknown')} ({product.get('country', 'Unknown')})")
