"""Extraction result models for tracking processing outcomes."""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
import uuid

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from .product import Product, ExtractionMetadata, ProcessingStatus
    from .visual_element import VisualElement
else:
    # Import only the types we need at runtime
    from .product import Product, ExtractionMetadata, ProcessingStatus
    from .visual_element import VisualElement

logger = logging.getLogger(__name__)


class ExtractionMethod(Enum):
    """Method used for extraction."""
    OPENAI_GPT = "openai_gpt"
    MANUAL = "manual"
    RULE_BASED = "rule_based"
    HYBRID = "hybrid"


@dataclass
class ExtractionResult:
    """Complete result of bioeconomic product extraction."""

    # Results
    products: List[Product] = field(default_factory=list)
    status: ProcessingStatus = ProcessingStatus.PENDING
    overall_confidence: float = 0.0

    # Visual elements
    visual_elements: List[VisualElement] = field(default_factory=list)

    # Metadata
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    metadata: Optional[ExtractionMetadata] = None

    # Summary statistics
    total_products_found: int = 0
    total_visual_elements: int = 0
    unique_countries: List[str] = field(default_factory=list)
    product_categories: Dict[str, int] = field(default_factory=dict)
    visual_element_types: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        """Update calculated fields after initialization."""
        self.update_statistics()

    def add_product(self, product: Product):
        """
        Add a product to the results.

        Args:
            product: Product to add
        """
        self.products.append(product)
        self.update_statistics()

    def add_products(self, products: List[Product]):
        """
        Add multiple products to the results.

        Args:
            products: List of products to add
        """
        self.products.extend(products)
        self.update_statistics()

    def add_visual_element(self, element: VisualElement):
        """
        Add a visual element to the results.

        Args:
            element: Visual element to add
        """
        self.visual_elements.append(element)
        self.update_statistics()

    def add_visual_elements(self, elements: List[VisualElement]):
        """
        Add multiple visual elements to the results.

        Args:
            elements: List of visual elements to add
        """
        self.visual_elements.extend(elements)
        self.update_statistics()

    def update_statistics(self):
        """Update summary statistics based on current products and visual elements."""
        self.total_products_found = len(self.products)
        self.total_visual_elements = len(self.visual_elements)

        # Update unique countries
        countries = set()
        for product in self.products:
            if product.origin and product.origin.country:
                countries.add(product.origin.country)
        self.unique_countries = sorted(list(countries))

        # Update product categories
        categories = {}
        for product in self.products:
            for use in product.uses:
                category = use.category.value
                categories[category] = categories.get(category, 0) + 1
        self.product_categories = categories

        # Update visual element types
        visual_types = {}
        for element in self.visual_elements:
            element_type = element.element_type.value
            visual_types[element_type] = visual_types.get(element_type, 0) + 1
        self.visual_element_types = visual_types

        # Update overall confidence
        if self.products:
            self.overall_confidence = sum(p.confidence_score for p in self.products) / len(self.products)
        else:
            self.overall_confidence = 0.0

    def mark_completed(self):
        """Mark the extraction as completed."""
        self.status = ProcessingStatus.COMPLETED
        self.completed_at = datetime.now()
        self.update_statistics()

    def mark_failed(self, error_message: str = ""):
        """
        Mark the extraction as failed.

        Args:
            error_message: Optional error message
        """
        self.status = ProcessingStatus.FAILED
        self.completed_at = datetime.now()
        if error_message and self.metadata:
            self.metadata.add_error(error_message)

    def filter_by_confidence(self, min_confidence: float) -> 'ExtractionResult':
        """
        Create a new result with products above confidence threshold.

        Args:
            min_confidence: Minimum confidence score

        Returns:
            New ExtractionResult with filtered products
        """
        filtered_products = [p for p in self.products if p.confidence_score >= min_confidence]

        new_result = ExtractionResult(
            products=filtered_products,
            status=self.status,
            id=str(uuid.uuid4()),
            metadata=self.metadata
        )
        new_result.update_statistics()
        return new_result

    def filter_by_country(self, countries: List[str]) -> 'ExtractionResult':
        """
        Filter products by country.

        Args:
            countries: List of countries to include

        Returns:
            New ExtractionResult with filtered products
        """
        countries_lower = [c.lower() for c in countries]
        filtered_products = []

        for product in self.products:
            if product.origin and product.origin.country.lower() in countries_lower:
                filtered_products.append(product)

        new_result = ExtractionResult(
            products=filtered_products,
            status=self.status,
            id=str(uuid.uuid4()),
            metadata=self.metadata
        )
        new_result.update_statistics()
        return new_result

    def get_products_by_category(self, category: str) -> List[Product]:
        """
        Get products that have uses in a specific category.

        Args:
            category: Category name to filter by

        Returns:
            List of products with uses in the category
        """
        matching_products = []
        for product in self.products:
            if any(use.category.value == category for use in product.uses):
                matching_products.append(product)
        return matching_products

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the extraction results.

        Returns:
            Dictionary with summary statistics
        """
        processing_time = None
        if self.metadata and self.metadata.processing_time_seconds:
            processing_time = self.metadata.processing_time_seconds

        return {
            "id": self.id,
            "status": self.status.value,
            "total_products": self.total_products_found,
            "total_visual_elements": self.total_visual_elements,
            "unique_countries": len(self.unique_countries),
            "countries": self.unique_countries,
            "product_categories": self.product_categories,
            "visual_element_types": self.visual_element_types,
            "overall_confidence": round(self.overall_confidence, 3),
            "processing_time_seconds": processing_time,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "has_errors": bool(self.metadata.errors_encountered if self.metadata else False),
            "has_warnings": bool(self.metadata.warnings if self.metadata else False)
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to complete dictionary representation."""
        return {
            "id": self.id,
            "products": [product.to_dict() for product in self.products],
            "visual_elements": [element.to_dict() for element in self.visual_elements],
            "status": self.status.value,
            "overall_confidence": self.overall_confidence,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "total_products_found": self.total_products_found,
            "total_visual_elements": self.total_visual_elements,
            "unique_countries": self.unique_countries,
            "product_categories": self.product_categories,
            "visual_element_types": self.visual_element_types
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtractionResult':
        """
        Create ExtractionResult from dictionary.

        Args:
            data: Dictionary with extraction result data

        Returns:
            ExtractionResult instance
        """
        # Convert products
        products = []
        for product_data in data.get("products", []):
            products.append(Product.from_dict(product_data))

        # Convert visual elements
        visual_elements = []
        for element_data in data.get("visual_elements", []):
            visual_elements.append(VisualElement.from_dict(element_data))

        # Convert metadata
        metadata = None
        if data.get("metadata"):
            metadata_data = data["metadata"].copy()
            if "extraction_method" in metadata_data:
                metadata_data["extraction_method"] = ExtractionMethod(metadata_data["extraction_method"])
            metadata = ExtractionMetadata(**metadata_data)

        # Handle timestamps
        created_at = datetime.now()
        completed_at = None

        if "created_at" in data:
            created_at = datetime.fromisoformat(data["created_at"].replace('Z', '+00:00'))
        if data.get("completed_at"):
            completed_at = datetime.fromisoformat(data["completed_at"].replace('Z', '+00:00'))

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            products=products,
            visual_elements=visual_elements,
            status=ProcessingStatus(data.get("status", "pending")),
            overall_confidence=data.get("overall_confidence", 0.0),
            created_at=created_at,
            completed_at=completed_at,
            metadata=metadata,
            total_products_found=data.get("total_products_found", len(products)),
            total_visual_elements=data.get("total_visual_elements", len(visual_elements)),
            unique_countries=data.get("unique_countries", []),
            product_categories=data.get("product_categories", {}),
            visual_element_types=data.get("visual_element_types", {})
        )

    def __str__(self) -> str:
        """String representation of the extraction result."""
        return f"ExtractionResult(products={len(self.products)}, status={self.status.value})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"ExtractionResult(id='{self.id}', products={len(self.products)}, confidence={self.overall_confidence:.3f})"
