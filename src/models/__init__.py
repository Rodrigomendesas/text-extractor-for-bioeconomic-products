"""Models package for bioeconomic product analysis."""

from .product import (
    Product,
    ProductOrigin,
    ProductUse,
    ProductCategory,
    ProcessingLevel,
    ExtractionMetadata,
    ProcessingStatus
)

from .extraction_result import ExtractionResult, ExtractionMethod
from .visual_element import VisualElement, VisualElementType

__all__ = [
    "Product",
    "ProductOrigin", 
    "ProductUse",
    "ProductCategory",
    "ProcessingLevel",
    "ExtractionResult",
    "ExtractionMethod",
    "ExtractionMetadata",
    "ProcessingStatus",
    "VisualElement",
    "VisualElementType"
]
