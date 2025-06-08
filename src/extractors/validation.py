"""Product validation and quality checking."""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import re

from .base_extractor import BaseExtractor, ExtractorResult
from config import settings, prompt_manager

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Container for validation results."""
    is_valid: bool
    confidence_score: float
    corrections: Dict[str, Any]
    reasoning: str
    original_product: Dict[str, Any]
    validated_product: Dict[str, Any]


class ProductValidator:
    """Validates and improves extracted product information."""

    def __init__(self, extractor: Optional[BaseExtractor] = None):
        """
        Initialize the product validator.

        Args:
            extractor: BaseExtractor instance for LLM validation (optional)
        """
        self.extractor = extractor
        self.logger = logging.getLogger(__name__)

        # Known bioeconomic products for validation
        self.known_products = {
            'cacao', 'cocoa', 'chocolate', 'vanilla', 'vainilla', 'coffee', 'café', 
            'quinoa', 'quinua', 'guayusa', 'açaí', 'acai', 'brazil nut', 'nuez de brasil',
            'brazil nuts', 'nueces de brasil', 'coffee', 'tea', 'té', 'mate', 'yerba mate',
            'stevia', 'lucuma', 'lúcuma', 'spirulina', 'moringa', 'chia', 'amaranth',
            'amaranto', 'maca', 'sacha inchi', 'camu camu', 'dragon fruit', 'pitahaya'
        }

        # Known countries for validation
        self.known_countries = {
            'ecuador', 'peru', 'perú', 'brazil', 'brasil', 'colombia', 'bolivia',
            'venezuela', 'argentina', 'chile', 'uruguay', 'paraguay', 'guyana',
            'suriname', 'french guiana', 'mexico', 'méxico', 'guatemala', 'belize',
            'honduras', 'el salvador', 'nicaragua', 'costa rica', 'panama', 'panamá'
        }

    def validate_product(self, product: Dict[str, Any], context: str = "") -> ValidationResult:
        """
        Validate a single product entry.

        Args:
            product: Product dictionary to validate
            context: Original text context for validation

        Returns:
            ValidationResult with validation outcome and corrections
        """
        self.logger.debug(f"Validating product: {product.get('product_name', 'Unknown')}")

        # Start with the original product
        validated_product = product.copy()
        corrections = {}
        issues = []

        # Validate product name
        name_result = self._validate_product_name(product.get('product_name', ''))
        if not name_result[0]:
            issues.append(name_result[1])
        elif name_result[2]:  # Has correction
            corrections['product_name'] = name_result[2]
            validated_product['product_name'] = name_result[2]

        # Validate country
        country_result = self._validate_country(product.get('country', ''))
        if not country_result[0]:
            issues.append(country_result[1])
        elif country_result[2]:  # Has correction
            corrections['country'] = country_result[2]
            validated_product['country'] = country_result[2]

        # Validate scientific name
        scientific_name = product.get('scientific_name', '')
        if scientific_name:
            scientific_result = self._validate_scientific_name(scientific_name)
            if not scientific_result[0]:
                issues.append(scientific_result[1])
            elif scientific_result[2]:  # Has correction
                corrections['scientific_name'] = scientific_result[2]
                validated_product['scientific_name'] = scientific_result[2]

        # Validate uses
        uses_result = self._validate_uses(product.get('uses', []))
        if not uses_result[0]:
            issues.append(uses_result[1])
        elif uses_result[2]:  # Has correction
            corrections['uses'] = uses_result[2]
            validated_product['uses'] = uses_result[2]

        # Calculate validation confidence
        confidence_score = self._calculate_validation_confidence(validated_product, issues)

        # Determine if valid
        is_valid = len(issues) == 0 and confidence_score > 0.6

        reasoning = f"Validation {'passed' if is_valid else 'failed'}."
        if issues:
            reasoning += f" Issues: {'; '.join(issues)}"
        if corrections:
            reasoning += f" Applied {len(corrections)} corrections."

        return ValidationResult(
            is_valid=is_valid,
            confidence_score=confidence_score,
            corrections=corrections,
            reasoning=reasoning,
            original_product=product,
            validated_product=validated_product
        )

    def validate_extraction_result(self, result: ExtractorResult, context: str = "") -> ExtractorResult:
        """
        Validate all products in an extraction result.

        Args:
            result: ExtractorResult to validate
            context: Original text context

        Returns:
            New ExtractorResult with validated products
        """
        self.logger.info(f"Validating {len(result.products)} extracted products")

        validated_products = []
        total_corrections = 0

        for product in result.products:
            validation_result = self.validate_product(product, context)

            if validation_result.is_valid or validation_result.confidence_score > 0.5:
                validated_products.append(validation_result.validated_product)
                total_corrections += len(validation_result.corrections)
            else:
                self.logger.warning(f"Rejected product: {product.get('product_name', 'Unknown')} - {validation_result.reasoning}")

        # Adjust confidence score based on validation
        validation_factor = len(validated_products) / len(result.products) if result.products else 0
        adjusted_confidence = result.confidence_score * validation_factor

        self.logger.info(f"Validation complete: {len(validated_products)}/{len(result.products)} products retained, {total_corrections} corrections applied")

        return ExtractorResult(
            products=validated_products,
            confidence_score=adjusted_confidence,
            extraction_method=result.extraction_method + "-validated",
            tokens_used=result.tokens_used,
            processing_time=result.processing_time,
            source_text_length=result.source_text_length
        )

    def _validate_product_name(self, name: str) -> Tuple[bool, str, Optional[str]]:
        """Validate product name. Returns (is_valid, error_message, correction)."""
        if not name or len(name.strip()) < 2:
            return False, "Product name too short or empty", None

        name = name.strip()
        name_lower = name.lower()

        # Check if it's a known bioeconomic product
        if any(known in name_lower for known in self.known_products):
            return True, "", None

        # Check for common patterns
        bio_keywords = ['organic', 'natural', 'extract', 'oil', 'seed', 'fruit', 'bean', 'nut', 'herb']
        if any(keyword in name_lower for keyword in bio_keywords):
            return True, "", None

        # Check for obvious non-products
        invalid_patterns = ['company', 'corporation', 'inc', 'ltd', 'spa', 'sa']
        if any(pattern in name_lower for pattern in invalid_patterns):
            return False, "Name appears to be a company rather than a product", None

        # If not clearly invalid, accept with lower confidence
        return True, "", None

    def _validate_country(self, country: str) -> Tuple[bool, str, Optional[str]]:
        """Validate country name. Returns (is_valid, error_message, correction)."""
        if not country or len(country.strip()) < 2:
            return False, "Country name too short or empty", None

        country = country.strip()
        country_lower = country.lower()

        # Direct match
        if country_lower in self.known_countries:
            return True, "", None

        # Common corrections
        corrections = {
            'brazil': 'Brazil',
            'brasil': 'Brazil', 
            'peru': 'Peru',
            'perú': 'Peru',
            'mexico': 'Mexico',
            'méxico': 'Mexico',
            'colombia': 'Colombia',
            'ecuador': 'Ecuador',
            'bolivia': 'Bolivia',
            'venezuela': 'Venezuela'
        }

        if country_lower in corrections:
            return True, "", corrections[country_lower]

        # If not recognized, still accept (might be valid but not in our list)
        return True, "", None

    def _validate_scientific_name(self, name: str) -> Tuple[bool, str, Optional[str]]:
        """Validate scientific name format. Returns (is_valid, error_message, correction)."""
        if not name:
            return True, "", None  # Empty is OK

        name = name.strip()

        # Basic format check: should be "Genus species" (two words, capitalized)
        pattern = r'^[A-Z][a-z]+ [a-z]+$'
        if re.match(pattern, name):
            return True, "", None

        # Try to fix common issues
        words = name.split()
        if len(words) >= 2:
            # Capitalize genus, lowercase species
            corrected = f"{words[0].capitalize()} {words[1].lower()}"
            if re.match(pattern, corrected):
                return True, "", corrected

        return False, "Scientific name format invalid (should be 'Genus species')", None

    def _validate_uses(self, uses: List[str]) -> Tuple[bool, str, Optional[List[str]]]:
        """Validate uses list. Returns (is_valid, error_message, correction)."""
        if not uses:
            return False, "No uses specified", None

        if not isinstance(uses, list):
            return False, "Uses should be a list", None

        # Clean up uses
        cleaned_uses = []
        for use in uses:
            if isinstance(use, str) and len(use.strip()) > 0:
                cleaned_uses.append(use.strip())

        if not cleaned_uses:
            return False, "No valid uses found", None

        return True, "", cleaned_uses if cleaned_uses != uses else None

    def _calculate_validation_confidence(self, product: Dict[str, Any], issues: List[str]) -> float:
        """Calculate confidence score for validated product."""
        score = 1.0

        # Penalize for issues
        score -= len(issues) * 0.2

        # Bonus for completeness
        if product.get('scientific_name'):
            score += 0.1
        if product.get('additional_info'):
            score += 0.1
        if len(product.get('uses', [])) > 1:
            score += 0.1

        # Bonus for recognized products/countries
        name_lower = product.get('product_name', '').lower()
        if any(known in name_lower for known in self.known_products):
            score += 0.2

        country_lower = product.get('country', '').lower()
        if country_lower in self.known_countries:
            score += 0.1

        return max(0.0, min(1.0, score))
