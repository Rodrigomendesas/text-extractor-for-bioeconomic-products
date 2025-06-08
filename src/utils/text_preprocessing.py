"""Text preprocessing utilities for bioeconomic products analysis."""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import string

logger = logging.getLogger(__name__)


@dataclass
class ProcessingMetrics:
    """Metrics from text preprocessing."""
    original_length: int
    processed_length: int
    reduction_ratio: float
    language_detected: Optional[str] = None
    confidence: float = 0.0
    chunks_created: int = 0
    content_type: Optional[str] = None


class LanguageDetector:
    """Simple language detection for text preprocessing."""
    
    def __init__(self):
        """Initialize language detector."""
        # Common words in different languages relevant to our domain
        self.language_patterns = {
            'spanish': {
                'keywords': [
                    'el', 'la', 'de', 'en', 'y', 'a', 'que', 'es', 'se', 'no', 'un', 'por', 'con', 'para',
                    'productos', 'planta', 'medicinal', 'uso', 'traditional', 'país', 'región'
                ],
                'indicators': ['ñ', 'á', 'é', 'í', 'ó', 'ú']
            },
            'portuguese': {
                'keywords': [
                    'o', 'a', 'de', 'e', 'do', 'da', 'em', 'um', 'para', 'com', 'não', 'uma', 'por', 'mais',
                    'produtos', 'planta', 'medicinal', 'uso', 'país', 'região', 'brasil'
                ],
                'indicators': ['ã', 'õ', 'ç', 'á', 'é', 'í', 'ó', 'ú']
            },
            'english': {
                'keywords': [
                    'the', 'of', 'and', 'a', 'to', 'in', 'is', 'it', 'you', 'that', 'he', 'was', 'for', 'on',
                    'products', 'plant', 'medicinal', 'use', 'traditional', 'country', 'region'
                ],
                'indicators': []
            },
            'french': {
                'keywords': [
                    'le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour', 'dans', 'ce',
                    'produits', 'plante', 'médicinal', 'usage', 'pays', 'région'
                ],
                'indicators': ['à', 'ç', 'é', 'è', 'ê', 'ë', 'î', 'ï', 'ô', 'ù', 'û', 'ü', 'ÿ']
            }
        }
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect language of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (language, confidence)
        """
        if not text or len(text) < 20:
            return 'unknown', 0.0
        
        text_lower = text.lower()
        scores = {}
        
        for language, patterns in self.language_patterns.items():
            score = 0
            
            # Count keyword matches
            for keyword in patterns['keywords']:
                # Count whole word matches
                count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', text_lower))
                score += count
            
            # Count character indicators
            for indicator in patterns['indicators']:
                score += text_lower.count(indicator) * 2  # Weight character indicators more
            
            scores[language] = score
        
        if not scores or max(scores.values()) == 0:
            return 'unknown', 0.0
        
        # Find best match
        best_language = max(scores, key=scores.get)
        max_score = scores[best_language]
        total_score = sum(scores.values())
        
        confidence = max_score / total_score if total_score > 0 else 0.0
        confidence = min(1.0, confidence)  # Cap at 1.0
        
        return best_language, confidence


class TextCleaner:
    """Clean and normalize text for processing."""
    
    def __init__(self, preserve_structure: bool = True):
        """
        Initialize text cleaner.
        
        Args:
            preserve_structure: Whether to preserve paragraph structure
        """
        self.preserve_structure = preserve_structure
    
    def clean(self, text: str) -> str:
        """
        Clean text while preserving important information.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove or replace problematic characters
        cleaned = self._remove_control_characters(text)
        cleaned = self._normalize_whitespace(cleaned)
        cleaned = self._fix_encoding_issues(cleaned)
        cleaned = self._normalize_punctuation(cleaned)
        
        if self.preserve_structure:
            cleaned = self._preserve_paragraph_breaks(cleaned)
        
        return cleaned.strip()
    
    def _remove_control_characters(self, text: str) -> str:
        """Remove control characters but preserve important ones."""
        # Keep newlines, tabs, and carriage returns
        allowed_controls = {'\n', '\t', '\r'}
        return ''.join(char for char in text if ord(char) >= 32 or char in allowed_controls)
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace while preserving structure."""
        if self.preserve_structure:
            # Normalize within lines but preserve line breaks
            lines = text.split('\n')
            normalized_lines = []
            for line in lines:
                # Replace multiple spaces/tabs with single space
                normalized_line = re.sub(r'[ \t]+', ' ', line.strip())
                normalized_lines.append(normalized_line)
            return '\n'.join(normalized_lines)
        else:
            # Replace all whitespace with single spaces
            return re.sub(r'\s+', ' ', text)
    
    def _fix_encoding_issues(self, text: str) -> str:
        """Fix common encoding issues."""
        replacements = {
            'â€™': "'",
            'â€œ': '"',
            'â€\x9d': '"',
            'â€"': '–',
            'â€"': '—',
            'Ã¡': 'á',
            'Ã©': 'é',
            'Ã­': 'í',
            'Ã³': 'ó',
            'Ãº': 'ú',
            'Ã±': 'ñ',
            'Ã§': 'ç'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def _normalize_punctuation(self, text: str) -> str:
        """Normalize punctuation marks."""
        # Replace different types of quotes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"['']", "'", text)
        
        # Replace different types of dashes
        text = re.sub(r'[–—]', '-', text)
        
        # Normalize ellipsis
        text = re.sub(r'\.{3,}', '...', text)
        
        return text
    
    def _preserve_paragraph_breaks(self, text: str) -> str:
        """Preserve paragraph breaks while cleaning."""
        # Convert multiple newlines to paragraph markers
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove single newlines within paragraphs (line wrapping)
        paragraphs = text.split('\n\n')
        cleaned_paragraphs = []
        
        for paragraph in paragraphs:
            # Remove single newlines within paragraph
            cleaned_para = re.sub(r'(?<!\n)\n(?!\n)', ' ', paragraph)
            cleaned_paragraphs.append(cleaned_para.strip())
        
        return '\n\n'.join(p for p in cleaned_paragraphs if p)


class TextChunker:
    """Intelligent text chunking for processing."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 100,
        respect_sentences: bool = True,
        respect_paragraphs: bool = True
    ):
        """
        Initialize text chunker.
        
        Args:
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks in characters
            respect_sentences: Try to break at sentence boundaries
            respect_paragraphs: Try to break at paragraph boundaries
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.respect_sentences = respect_sentences
        self.respect_paragraphs = respect_paragraphs
    
    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into intelligent chunks.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of chunk dictionaries with metadata
        """
        if len(text) <= self.chunk_size:
            return [{
                'text': text,
                'start_pos': 0,
                'end_pos': len(text),
                'chunk_id': 0,
                'size': len(text)
            }]
        
        chunks = []
        
        if self.respect_paragraphs:
            chunks = self._chunk_by_paragraphs(text)
        else:
            chunks = self._chunk_by_sentences(text) if self.respect_sentences else self._chunk_by_size(text)
        
        # Add metadata to chunks
        for i, chunk in enumerate(chunks):
            chunk['chunk_id'] = i
            chunk['size'] = len(chunk['text'])
        
        return chunks
    
    def _chunk_by_paragraphs(self, text: str) -> List[Dict[str, Any]]:
        """Chunk text respecting paragraph boundaries."""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        start_pos = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If adding this paragraph exceeds chunk size
            if current_chunk and len(current_chunk) + len(paragraph) + 2 > self.chunk_size:
                # Save current chunk
                chunks.append({
                    'text': current_chunk.strip(),
                    'start_pos': start_pos,
                    'end_pos': start_pos + len(current_chunk)
                })
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + paragraph
                start_pos = start_pos + len(current_chunk) - len(overlap_text)
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += '\n\n' + paragraph
                else:
                    current_chunk = paragraph
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                'text': current_chunk.strip(),
                'start_pos': start_pos,
                'end_pos': start_pos + len(current_chunk)
            })
        
        return chunks
    
    def _chunk_by_sentences(self, text: str) -> List[Dict[str, Any]]:
        """Chunk text respecting sentence boundaries."""
        sentences = self._split_sentences(text)
        chunks = []
        current_chunk = ""
        start_pos = 0
        
        for sentence in sentences:
            if current_chunk and len(current_chunk) + len(sentence) > self.chunk_size:
                # Save current chunk
                chunks.append({
                    'text': current_chunk.strip(),
                    'start_pos': start_pos,
                    'end_pos': start_pos + len(current_chunk)
                })
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + sentence
                start_pos = start_pos + len(current_chunk) - len(overlap_text)
            else:
                current_chunk += sentence
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                'text': current_chunk.strip(),
                'start_pos': start_pos,
                'end_pos': start_pos + len(current_chunk)
            })
        
        return chunks
    
    def _chunk_by_size(self, text: str) -> List[Dict[str, Any]]:
        """Simple size-based chunking with overlap."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            if end >= len(text):
                chunk_text = text[start:]
            else:
                chunk_text = text[start:end]
            
            chunks.append({
                'text': chunk_text,
                'start_pos': start,
                'end_pos': start + len(chunk_text)
            })
            
            start = end - self.overlap
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s for s in sentences if s.strip()]
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from end of chunk."""
        if len(text) <= self.overlap:
            return text
        
        overlap_text = text[-self.overlap:]
        
        # Try to start at a word boundary
        space_pos = overlap_text.find(' ')
        if space_pos > 0:
            overlap_text = overlap_text[space_pos:].strip()
        
        return overlap_text + ' ' if overlap_text else ''


class ContentFilter:
    """Filter content to identify relevant bioeconomic information."""
    
    def __init__(self):
        """Initialize content filter."""
        # Keywords that indicate bioeconomic content
        self.relevant_keywords = {
            'products': [
                'plant', 'plants', 'medicinal', 'traditional', 'natural', 'biological',
                'bioeconomic', 'biodiversity', 'species', 'extract', 'oil', 'fiber',
                'medicine', 'food', 'cosmetic', 'industrial', 'economic', 'commercial'
            ],
            'usage': [
                'use', 'used', 'application', 'treatment', 'therapy', 'remedy',
                'preparation', 'processing', 'harvest', 'cultivation', 'production'
            ],
            'geography': [
                'amazon', 'rainforest', 'forest', 'tropical', 'native', 'indigenous',
                'region', 'area', 'zone', 'habitat', 'ecosystem', 'conservation'
            ]
        }
        
        # Compile patterns for efficiency
        self.keyword_patterns = {}
        for category, keywords in self.relevant_keywords.items():
            pattern = r'\b(?:' + '|'.join(re.escape(kw) for kw in keywords) + r')\b'
            self.keyword_patterns[category] = re.compile(pattern, re.IGNORECASE)
    
    def calculate_relevance_score(self, text: str) -> float:
        """
        Calculate relevance score for bioeconomic content.
        
        Args:
            text: Text to analyze
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        if not text:
            return 0.0
        
        text_lower = text.lower()
        total_words = len(text.split())
        
        if total_words == 0:
            return 0.0
        
        score = 0.0
        
        # Score based on keyword categories
        for category, pattern in self.keyword_patterns.items():
            matches = len(pattern.findall(text))
            category_score = min(matches / total_words * 100, 1.0)  # Cap at 1.0
            
            # Weight different categories
            if category == 'products':
                score += category_score * 0.5
            elif category == 'usage':
                score += category_score * 0.3
            elif category == 'geography':
                score += category_score * 0.2
        
        # Bonus for scientific names (Genus species pattern)
        scientific_names = re.findall(r'\b[A-Z][a-z]+ [a-z]+\b', text)
        if scientific_names:
            score += min(len(scientific_names) / total_words * 50, 0.3)
        
        # Bonus for country names in our region
        from .helpers import extract_countries
        countries = extract_countries(text, region_filter=True)
        if countries:
            score += min(len(countries) / 10, 0.2)
        
        return min(score, 1.0)
    
    def is_relevant(self, text: str, threshold: float = 0.1) -> bool:
        """
        Check if text is relevant for bioeconomic analysis.
        
        Args:
            text: Text to check
            threshold: Minimum relevance score
            
        Returns:
            True if text is relevant
        """
        return self.calculate_relevance_score(text) >= threshold
    
    def extract_relevant_sections(self, text: str, threshold: float = 0.1) -> List[str]:
        """
        Extract relevant sections from text.
        
        Args:
            text: Text to analyze
            threshold: Minimum relevance score for sections
            
        Returns:
            List of relevant text sections
        """
        # Split into paragraphs
        paragraphs = text.split('\n\n')
        relevant_sections = []
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if paragraph and self.is_relevant(paragraph, threshold):
                relevant_sections.append(paragraph)
        
        return relevant_sections


class TextPreprocessor:
    """Main text preprocessing coordinator."""
    
    def __init__(
        self,
        clean_text: bool = True,
        detect_language: bool = True,
        filter_content: bool = True,
        chunk_text: bool = True,
        chunk_size: int = 1000,
        overlap: int = 100
    ):
        """
        Initialize text preprocessor.
        
        Args:
            clean_text: Whether to clean text
            detect_language: Whether to detect language
            filter_content: Whether to filter for relevant content
            chunk_text: Whether to chunk text
            chunk_size: Size of text chunks
            overlap: Overlap between chunks
        """
        self.clean_text_enabled = clean_text
        self.detect_language_enabled = detect_language
        self.filter_content_enabled = filter_content
        self.chunk_text_enabled = chunk_text
        
        # Initialize components
        self.cleaner = TextCleaner() if clean_text else None
        self.language_detector = LanguageDetector() if detect_language else None
        self.content_filter = ContentFilter() if filter_content else None
        self.chunker = TextChunker(chunk_size=chunk_size, overlap=overlap) if chunk_text else None
    
    def process(self, text: str) -> Dict[str, Any]:
        """
        Process text through all preprocessing steps.
        
        Args:
            text: Text to process
            
        Returns:
            Dictionary with processed text and metadata
        """
        original_length = len(text)
        processed_text = text
        metrics = ProcessingMetrics(original_length=original_length, processed_length=0, reduction_ratio=0.0)
        
        # Clean text
        if self.clean_text_enabled and self.cleaner:
            processed_text = self.cleaner.clean(processed_text)
        
        # Detect language
        if self.detect_language_enabled and self.language_detector:
            language, confidence = self.language_detector.detect_language(processed_text)
            metrics.language_detected = language
            metrics.confidence = confidence
        
        # Filter content
        relevant_sections = []
        relevance_score = 0.0
        if self.filter_content_enabled and self.content_filter:
            relevance_score = self.content_filter.calculate_relevance_score(processed_text)
            if relevance_score > 0.1:
                relevant_sections = self.content_filter.extract_relevant_sections(processed_text)
            metrics.content_type = "relevant" if relevance_score > 0.1 else "low_relevance"
        
        # Chunk text
        chunks = []
        if self.chunk_text_enabled and self.chunker:
            chunks = self.chunker.chunk_text(processed_text)
            metrics.chunks_created = len(chunks)
        
        # Update metrics
        metrics.processed_length = len(processed_text)
        metrics.reduction_ratio = (original_length - metrics.processed_length) / original_length if original_length > 0 else 0.0
        
        return {
            'original_text': text,
            'processed_text': processed_text,
            'chunks': chunks,
            'relevant_sections': relevant_sections,
            'relevance_score': relevance_score,
            'metrics': metrics
        }