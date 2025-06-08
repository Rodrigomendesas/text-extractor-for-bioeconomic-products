"""PDF processing orchestrator that combines text extraction and language detection."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .text_extractor import TextExtractor
from .language_detector import LanguageDetector
from config import settings
from src.models.visual_element import VisualElement

logger = logging.getLogger(__name__)


@dataclass
class ProcessedDocument:
    """Container for processed document information."""
    file_path: Path
    text: str
    language: Dict[str, str]
    chunks: List[str]
    page_count: int
    extraction_method: str
    char_count: int
    visual_elements: List[VisualElement] = None


class PDFProcessor:
    """High-level PDF processor that orchestrates text extraction and language detection."""

    def __init__(self):
        """Initialize the PDF processor with its dependencies."""
        self.text_extractor = TextExtractor()
        self.language_detector = LanguageDetector()
        self.chunk_size = settings.max_chunk_size

    def process_pdf(self, pdf_path: Path) -> ProcessedDocument:
        """
        Process a PDF file completely: extract text and visual elements, detect language, and chunk.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            ProcessedDocument with all extracted information
        """
        logger.info(f"Processing PDF: {pdf_path}")

        # Extract text and visual elements
        extraction_result = self.text_extractor.extract_from_file(pdf_path)
        text = extraction_result['text']
        visual_elements = extraction_result.get('visual_elements', [])

        if not text.strip() and not visual_elements:
            raise ValueError(f"No content could be extracted from {pdf_path}")

        # Detect language
        language = self.language_detector.detect_language(text)
        logger.info(f"Detected language: {language['name']} ({language['code']})")

        # Create text chunks
        chunks = self.text_extractor.chunk_text(text, self.chunk_size)
        logger.info(f"Created {len(chunks)} text chunks")

        # Log visual elements
        if visual_elements:
            logger.info(f"Extracted {len(visual_elements)} visual elements")

            # Count by type
            type_counts = {}
            for element in visual_elements:
                element_type = element.element_type.value
                type_counts[element_type] = type_counts.get(element_type, 0) + 1

            for element_type, count in type_counts.items():
                logger.info(f"  - {element_type}: {count}")

        return ProcessedDocument(
            file_path=pdf_path,
            text=text,
            language=language,
            chunks=chunks,
            page_count=extraction_result['page_count'],
            extraction_method=extraction_result['method'],
            char_count=len(text),
            visual_elements=visual_elements
        )

    def process_directory(self, directory_path: Path) -> List[ProcessedDocument]:
        """
        Process all PDF files in a directory.

        Args:
            directory_path: Path to directory containing PDFs

        Returns:
            List of ProcessedDocument objects
        """
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        pdf_files = list(directory_path.glob("*.pdf"))

        if not pdf_files:
            logger.warning(f"No PDF files found in {directory_path}")
            return []

        logger.info(f"Found {len(pdf_files)} PDF files to process")

        processed_docs = []
        for pdf_file in pdf_files:
            try:
                doc = self.process_pdf(pdf_file)
                processed_docs.append(doc)
                logger.info(f"Successfully processed: {pdf_file.name}")
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {e}")
                continue

        logger.info(f"Successfully processed {len(processed_docs)} out of {len(pdf_files)} PDFs")
        return processed_docs

    def get_processing_stats(self, documents: List[ProcessedDocument]) -> Dict[str, Any]:
        """
        Get statistics about processed documents.

        Args:
            documents: List of processed documents

        Returns:
            Dictionary with processing statistics
        """
        if not documents:
            return {}

        total_chars = sum(doc.char_count for doc in documents)
        total_pages = sum(doc.page_count for doc in documents)
        total_chunks = sum(len(doc.chunks) for doc in documents)
        total_visual_elements = sum(len(doc.visual_elements or []) for doc in documents)

        languages = {}
        methods = {}
        visual_element_types = {}

        for doc in documents:
            # Language stats
            lang = doc.language['code']
            languages[lang] = languages.get(lang, 0) + 1

            # Extraction method stats
            method = doc.extraction_method
            methods[method] = methods.get(method, 0) + 1

            # Visual element stats
            if doc.visual_elements:
                for element in doc.visual_elements:
                    element_type = element.element_type.value
                    visual_element_types[element_type] = visual_element_types.get(element_type, 0) + 1

        return {
            'total_documents': len(documents),
            'total_pages': total_pages,
            'total_characters': total_chars,
            'total_chunks': total_chunks,
            'total_visual_elements': total_visual_elements,
            'average_chars_per_doc': total_chars // len(documents),
            'languages_detected': languages,
            'extraction_methods_used': methods,
            'visual_element_types': visual_element_types
        }
