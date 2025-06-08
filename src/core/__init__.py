"""Core processing modules for bioeconomic products analysis."""

from .language_detector import LanguageDetector
from .pdf_processor import PDFProcessor
from .text_extractor import TextExtractor

__all__ = [
    "LanguageDetector",
    "PDFProcessor", 
    "TextExtractor"
]