"""Models for representing visual elements extracted from documents."""

import logging
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import uuid
from datetime import datetime
import base64

logger = logging.getLogger(__name__)


class VisualElementType(Enum):
    """Types of visual elements that can be extracted."""
    IMAGE = "image"
    CHART = "chart"
    TABLE = "table"
    DIAGRAM = "diagram"
    UNKNOWN = "unknown"


@dataclass
class VisualElement:
    """Represents a visual element extracted from a document."""
    
    # Basic information
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    element_type: VisualElementType = VisualElementType.UNKNOWN
    page_number: int = 0
    position: Tuple[float, float, float, float] = (0, 0, 0, 0)  # x0, y0, x1, y1
    
    # Content
    content_base64: Optional[str] = None  # Base64 encoded image data
    content_format: str = "png"  # Format of the image (png, jpg, etc.)
    
    # Metadata
    width: int = 0
    height: int = 0
    dpi: int = 0
    extracted_at: datetime = field(default_factory=datetime.now)
    
    # Analysis
    caption: Optional[str] = None
    description: Optional[str] = None
    ocr_text: Optional[str] = None
    confidence_score: float = 0.0
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "element_type": self.element_type.value,
            "page_number": self.page_number,
            "position": self.position,
            "content_base64": self.content_base64,
            "content_format": self.content_format,
            "width": self.width,
            "height": self.height,
            "dpi": self.dpi,
            "extracted_at": self.extracted_at.isoformat(),
            "caption": self.caption,
            "description": self.description,
            "ocr_text": self.ocr_text,
            "confidence_score": self.confidence_score,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VisualElement':
        """Create VisualElement from dictionary."""
        # Handle timestamps
        extracted_at = datetime.now()
        if "extracted_at" in data:
            extracted_at = datetime.fromisoformat(data["extracted_at"].replace('Z', '+00:00'))
        
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            element_type=VisualElementType(data.get("element_type", "unknown")),
            page_number=data.get("page_number", 0),
            position=tuple(data.get("position", (0, 0, 0, 0))),
            content_base64=data.get("content_base64"),
            content_format=data.get("content_format", "png"),
            width=data.get("width", 0),
            height=data.get("height", 0),
            dpi=data.get("dpi", 0),
            extracted_at=extracted_at,
            caption=data.get("caption"),
            description=data.get("description"),
            ocr_text=data.get("ocr_text"),
            confidence_score=data.get("confidence_score", 0.0),
            tags=data.get("tags", [])
        )
    
    def __str__(self) -> str:
        """String representation of the visual element."""
        return f"VisualElement(type={self.element_type.value}, page={self.page_number})"