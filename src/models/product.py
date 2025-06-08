"""Models package for bioeconomic product analysis."""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid


class ProductCategory(Enum):
    """Categories for product uses."""
    FOOD = "food"
    MEDICINE = "medicine"
    COSMETIC = "cosmetic"
    CRAFT = "craft"
    CONSTRUCTION = "construction"
    TEXTILE = "textile"
    DYE = "dye"
    FUEL = "fuel"
    OTHER = "other"


class ProcessingLevel(Enum):
    """Processing level of the product."""
    RAW = "raw"
    SEMI_PROCESSED = "semi_processed"
    PROCESSED = "processed"


@dataclass
class ProductOrigin:
    """Geographic origin information for a product."""
    country: str
    region: Optional[str] = None
    locality: Optional[str] = None
    coordinates: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "country": self.country,
            "region": self.region,
            "locality": self.locality,
            "coordinates": self.coordinates
        }


@dataclass
class ProductUse:
    """Specific use of a product."""
    category: ProductCategory
    description: str
    traditional: bool = False
    commercial: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "description": self.description,
            "traditional": self.traditional,
            "commercial": self.commercial
        }


@dataclass
class ExtractionMetadata:
    """Metadata for extraction process."""
    source_file: str
    extraction_date: datetime = field(default_factory=datetime.now)
    model_used: Optional[str] = None
    confidence_threshold: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_file": self.source_file,
            "extraction_date": self.extraction_date.isoformat(),
            "model_used": self.model_used,
            "confidence_threshold": self.confidence_threshold
        }


# ExtractionResult moved to extraction_result.py to avoid circular imports


class ProcessingStatus(Enum):
    """Status of document processing."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Product:
    """Main product model for bioeconomic products."""
    product_name: str
    scientific_name: Optional[str] = None
    common_names: List[str] = field(default_factory=list)
    origin: Optional[ProductOrigin] = None
    uses: List[ProductUse] = field(default_factory=list)
    processing_level: ProcessingLevel = ProcessingLevel.RAW
    additional_info: str = ""

    # Metadata
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    confidence_score: float = 0.0
    source_document: Optional[str] = None
    extraction_method: Optional[str] = None

    def __post_init__(self):
        """Validate and clean product data."""
        # Clean required fields
        self.product_name = self.product_name.strip() if self.product_name else ""
        if not self.product_name:
            raise ValueError("Product name is required")

        # Clean optional fields
        if self.scientific_name:
            self.scientific_name = self.scientific_name.strip()
            self._validate_scientific_name()

        if self.additional_info:
            self.additional_info = self.additional_info.strip()

        # Clean common names
        self.common_names = [name.strip() for name in self.common_names if name.strip()]

        # Validate confidence score
        self.confidence_score = max(0.0, min(1.0, self.confidence_score))

    def _validate_scientific_name(self):
        """Validate scientific name format."""
        if not self.scientific_name:
            return

        # Basic validation: should be "Genus species" format
        parts = self.scientific_name.split()
        if len(parts) >= 2:
            # Capitalize genus, lowercase species
            self.scientific_name = f"{parts[0].capitalize()} {parts[1].lower()}"
            if len(parts) > 2:
                # Add any additional parts (subspecies, etc.)
                additional = " ".join(parts[2:]).lower()
                self.scientific_name += f" {additional}"

    def add_use(self, category: ProductCategory, description: str, **kwargs):
        """
        Add a new use to the product.

        Args:
            category: Use category
            description: Use description
            **kwargs: Additional use parameters
        """
        use = ProductUse(category=category, description=description, **kwargs)
        self.uses.append(use)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "product_name": self.product_name,
            "scientific_name": self.scientific_name,
            "common_names": self.common_names,
            "origin": self.origin.to_dict() if self.origin else None,
            "uses": [use.to_dict() for use in self.uses],
            "processing_level": self.processing_level.value,
            "additional_info": self.additional_info,
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "confidence_score": self.confidence_score,
            "source_document": self.source_document,
            "extraction_method": self.extraction_method
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Product':
        """
        Create a Product from a dictionary.

        Args:
            data: Dictionary with product data

        Returns:
            Product instance
        """
        # Create origin if present
        origin = None
        if data.get("origin"):
            origin = ProductOrigin(
                country=data["origin"].get("country", ""),
                region=data["origin"].get("region"),
                locality=data["origin"].get("locality")
            )

            if "coordinates" in data["origin"]:
                origin.coordinates = data["origin"]["coordinates"]

        # Create uses if present
        uses = []
        for use_data in data.get("uses", []):
            category = ProductCategory(use_data.get("category", "other"))
            use = ProductUse(
                category=category,
                description=use_data.get("description", ""),
                traditional=use_data.get("traditional_use", False),
                commercial=use_data.get("commercial_use", False)
            )
            uses.append(use)

        # Get processing level
        processing_level = ProcessingLevel.RAW
        if data.get("processing_level"):
            try:
                processing_level = ProcessingLevel(data["processing_level"])
            except ValueError:
                pass

        # Create product
        product = cls(
            product_name=data.get("product_name", ""),
            scientific_name=data.get("scientific_name"),
            common_names=data.get("common_names", []),
            origin=origin,
            uses=uses,
            processing_level=processing_level,
            additional_info=data.get("additional_info", ""),
            confidence_score=data.get("confidence_score", 0.0)
        )

        # Set metadata if present
        if "id" in data:
            product.id = data["id"]
        if "source_document" in data:
            product.source_document = data["source_document"]
        if "extraction_method" in data:
            product.extraction_method = data["extraction_method"]

        return product

    @classmethod
    def from_extraction_dict(cls, data: Dict[str, Any], confidence_score: float = 0.0, source_document: Optional[str] = None) -> 'Product':
        """
        Create a Product from a simplified extraction dictionary.

        Args:
            data: Dictionary with extracted product data
            confidence_score: Confidence score for the extraction
            source_document: Source document name

        Returns:
            Product instance
        """
        # Create product with basic info
        product = cls(
            product_name=data.get("product_name", ""),
            scientific_name=data.get("scientific_name"),
            additional_info=data.get("additional_info", ""),
            confidence_score=confidence_score,
            source_document=source_document
        )

        # Set origin if country is present
        if "country" in data:
            product.origin = ProductOrigin(
                country=data["country"],
                region=data.get("region")
            )

        # Add uses
        if "uses" in data:
            for use_text in data["uses"]:
                category = cls._categorize_use(use_text)
                product.add_use(category, use_text)

        return product

    @staticmethod
    def _categorize_use(use_text: str) -> ProductCategory:
        """
        Categorize a use based on its description.

        Args:
            use_text: Use description

        Returns:
            ProductCategory
        """
        use_lower = use_text.lower()

        # Check for food-related terms
        if any(term in use_lower for term in ["food", "eat", "consum", "nutri", "fruit", "veget"]):
            return ProductCategory.FOOD

        # Check for medicine-related terms
        if any(term in use_lower for term in ["medic", "heal", "treat", "cure", "remedy"]):
            return ProductCategory.MEDICINE

        # Check for cosmetic-related terms
        if any(term in use_lower for term in ["cosmet", "skin", "beauty", "hair"]):
            return ProductCategory.COSMETIC

        # Default to OTHER
        return ProductCategory.OTHER


# Export all models
__all__ = [
    "Product",
    "ProductOrigin",
    "ProductUse", 
    "ProductCategory",
    "ProcessingLevel",
    "ExtractionMetadata",
    "ProcessingStatus"
]
