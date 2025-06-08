"""Language detection functionality for text analysis."""

from typing import Dict, Optional
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import logging

# Set seed for consistent results
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)


class LanguageDetector:
    """Detects the language of text content."""
    
    SUPPORTED_LANGUAGES = {
        'en': 'English',
        'es': 'Spanish',
        'pt': 'Portuguese',
        'fr': 'French'
    }
    
    def __init__(self):
        """Initialize the language detector."""
        self.default_language = 'en'
    
    def detect_language(self, text: str) -> Dict[str, str]:
        """
        Detect the language of the given text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with 'code' and 'name' of detected language
        """
        if not text or len(text.strip()) < 10:
            logger.warning("Text too short for reliable language detection")
            return {
                'code': self.default_language,
                'name': self.SUPPORTED_LANGUAGES[self.default_language]
            }
        
        try:
            # Clean text for better detection
            cleaned_text = self._clean_text(text)
            detected_code = detect(cleaned_text)
            
            # Check if detected language is supported
            if detected_code in self.SUPPORTED_LANGUAGES:
                return {
                    'code': detected_code,
                    'name': self.SUPPORTED_LANGUAGES[detected_code]
                }
            else:
                logger.info(f"Detected unsupported language: {detected_code}, using default")
                return {
                    'code': self.default_language,
                    'name': self.SUPPORTED_LANGUAGES[self.default_language]
                }
                
        except LangDetectException as e:
            logger.warning(f"Language detection failed: {e}, using default")
            return {
                'code': self.default_language,
                'name': self.SUPPORTED_LANGUAGES[self.default_language]
            }
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text for better language detection.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace and newlines
        cleaned = ' '.join(text.split())
        
        # Take a sample if text is very long (language detection works better on shorter texts)
        if len(cleaned) > 1000:
            # Take text from middle to avoid headers/footers
            start = len(cleaned) // 4
            end = start + 1000
            cleaned = cleaned[start:end]
        
        return cleaned
    
    def is_supported_language(self, language_code: str) -> bool:
        """
        Check if a language code is supported.
        
        Args:
            language_code: ISO language code to check
            
        Returns:
            True if language is supported, False otherwise
        """
        return language_code in self.SUPPORTED_LANGUAGES