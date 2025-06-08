"""General utility functions and helpers."""

import logging
import hashlib
import time
import re
import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable, Union
from functools import wraps
import difflib

# Country patterns for extraction
LATIN_AMERICA_CARIBBEAN_COUNTRIES = {
    'argentina', 'bolivia', 'brazil', 'chile', 'colombia', 'ecuador', 'guyana',
    'paraguay', 'peru', 'suriname', 'uruguay', 'venezuela', 'french guiana',
    'antigua and barbuda', 'bahamas', 'barbados', 'belize', 'costa rica',
    'cuba', 'dominica', 'dominican republic', 'el salvador', 'grenada',
    'guatemala', 'haiti', 'honduras', 'jamaica', 'mexico', 'nicaragua',
    'panama', 'saint kitts and nevis', 'saint lucia', 'saint vincent and the grenadines',
    'trinidad and tobago', 'aruba', 'curacao', 'sint maarten', 'bonaire',
    'martinique', 'guadeloupe', 'puerto rico'
}


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        format_string: Custom format string
        
    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure logging
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=numeric_level,
        format=format_string,
        handlers=handlers,
        force=True
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized at level {log_level}")
    
    return logger


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_hash(file_path: Path, algorithm: str = "md5") -> str:
    """
    Calculate hash of a file.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm (md5, sha1, sha256)
        
    Returns:
        Hex digest of file hash
    """
    hash_obj = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 100,
    preserve_sentences: bool = True
) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to chunk
        chunk_size: Target size of each chunk
        overlap: Number of characters to overlap between chunks
        preserve_sentences: Try to break at sentence boundaries
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        if end >= len(text):
            # Last chunk
            chunks.append(text[start:])
            break
        
        # Try to find a good break point
        if preserve_sentences:
            # Look for sentence endings within the last 200 characters
            search_start = max(end - 200, start)
            sentence_ends = []
            
            for match in re.finditer(r'[.!?]\s+', text[search_start:end]):
                sentence_ends.append(search_start + match.end())
            
            if sentence_ends:
                end = sentence_ends[-1]
        
        chunks.append(text[start:end])
        start = end - overlap
    
    return chunks


def validate_file_path(file_path: Union[str, Path], must_exist: bool = True) -> bool:
    """
    Validate file path.
    
    Args:
        file_path: Path to validate
        must_exist: Whether file must exist
        
    Returns:
        True if path is valid
    """
    try:
        path = Path(file_path)
        
        if must_exist and not path.exists():
            return False
        
        if path.exists() and not path.is_file():
            return False
        
        # Check if parent directory exists or can be created
        if not path.parent.exists():
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError):
                return False
        
        return True
        
    except (OSError, ValueError):
        return False


def format_confidence_score(score: float, format_type: str = "percentage") -> str:
    """
    Format confidence score for display.
    
    Args:
        score: Confidence score (0.0 to 1.0)
        format_type: Format type ('percentage', 'decimal', 'stars')
        
    Returns:
        Formatted score string
    """
    if format_type == "percentage":
        return f"{score * 100:.1f}%"
    elif format_type == "decimal":
        return f"{score:.3f}"
    elif format_type == "stars":
        stars = int(score * 5)
        return "★" * stars + "☆" * (5 - stars)
    else:
        return str(score)


def clean_text(text: str) -> str:
    """
    Basic text cleaning.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def extract_countries(text: str, region_filter: bool = True) -> List[str]:
    """
    Extract country names from text.
    
    Args:
        text: Text to search for countries
        region_filter: Whether to filter for Latin America/Caribbean only
        
    Returns:
        List of found country names
    """
    text_lower = text.lower()
    found_countries = []
    
    countries_to_search = LATIN_AMERICA_CARIBBEAN_COUNTRIES if region_filter else set()
    
    # Add some common variations and full names
    country_variations = {
        'brazil': ['brasil', 'brazil'],
        'colombia': ['colombia', 'kolumbien'],
        'venezuela': ['venezuela', 'vzla'],
        'dominican republic': ['dominican republic', 'república dominicana', 'dom rep'],
        'costa rica': ['costa rica', 'costarica'],
        'el salvador': ['el salvador', 'salvador'],
        'trinidad and tobago': ['trinidad and tobago', 'trinidad', 'tobago'],
        'saint lucia': ['saint lucia', 'st lucia', 'st. lucia'],
        'saint kitts and nevis': ['saint kitts and nevis', 'st kitts', 'st. kitts'],
        'saint vincent and the grenadines': ['saint vincent', 'st vincent', 'st. vincent']
    }
    
    if region_filter:
        search_dict = country_variations
    else:
        # Add more countries if not filtering by region
        search_dict = country_variations.copy()
        # Could add more countries here
    
    # Search for countries
    for country, variants in search_dict.items():
        for variant in variants:
            if variant in text_lower:
                if country not in found_countries:
                    found_countries.append(country.title())
                break
    
    # Also search for exact matches from the main set
    for country in countries_to_search:
        if country in text_lower and country.title() not in found_countries:
            found_countries.append(country.title())
    
    return found_countries


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two texts using sequence matching.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not text1 or not text2:
        return 0.0
    
    # Normalize texts
    text1 = clean_text(text1.lower())
    text2 = clean_text(text2.lower())
    
    # Use sequence matcher
    matcher = difflib.SequenceMatcher(None, text1, text2)
    return matcher.ratio()


def retry_with_backoff(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for delay between retries
        exceptions: Tuple of exceptions to catch and retry
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        delay = backoff_factor ** attempt
                        time.sleep(delay)
                        continue
                    else:
                        break
            
            # If we get here, all retries failed
            raise last_exception
        
        return wrapper
    return decorator


def normalize_product_name(name: str) -> str:
    """
    Normalize product name for comparison.
    
    Args:
        name: Product name to normalize
        
    Returns:
        Normalized product name
    """
    if not name:
        return ""
    
    # Convert to lowercase
    normalized = name.lower()
    
    # Remove common prefixes/suffixes
    prefixes = ['extract of', 'oil of', 'powder of', 'dried', 'fresh']
    suffixes = ['extract', 'oil', 'powder', 'leaves', 'bark', 'root', 'seeds']
    
    for prefix in prefixes:
        if normalized.startswith(prefix + ' '):
            normalized = normalized[len(prefix) + 1:]
    
    for suffix in suffixes:
        if normalized.endswith(' ' + suffix):
            normalized = normalized[:-len(suffix) - 1]
    
    # Remove extra whitespace and punctuation
    normalized = re.sub(r'[^\w\s]', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized)
    normalized = normalized.strip()
    
    return normalized


def validate_scientific_name(scientific_name: str) -> bool:
    """
    Validate scientific name format.
    
    Args:
        scientific_name: Scientific name to validate
        
    Returns:
        True if format appears valid
    """
    if not scientific_name:
        return False
    
    # Basic validation: should have at least genus and species
    parts = scientific_name.strip().split()
    
    if len(parts) < 2:
        return False
    
    # Genus should start with capital letter
    if not parts[0][0].isupper():
        return False
    
    # Species should be lowercase
    if not parts[1].islower():
        return False
    
    # Should contain only letters and common scientific name characters
    allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .-')
    if not all(c in allowed_chars for c in scientific_name):
        return False
    
    return True


def estimate_processing_time(text_length: int, method: str = "llm") -> float:
    """
    Estimate processing time based on text length and method.
    
    Args:
        text_length: Length of text in characters
        method: Processing method ('llm', 'rule_based', 'manual')
        
    Returns:
        Estimated time in seconds
    """
    if method == "llm":
        # Rough estimate: 1000 characters per 2 seconds for LLM processing
        return (text_length / 1000) * 2.0
    elif method == "rule_based":
        # Much faster for rule-based
        return (text_length / 10000) * 1.0
    else:
        # Manual processing is much slower
        return (text_length / 100) * 60.0


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for logging/debugging.
    
    Returns:
        Dictionary with system information
    """
    import platform
    import psutil
    
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "disk_free_gb": round(psutil.disk_usage('/').free / (1024**3), 2)
    }


def safe_filename(filename: str, max_length: int = 255) -> str:
    """
    Create a safe filename by removing/replacing problematic characters.
    
    Args:
        filename: Original filename
        max_length: Maximum filename length
        
    Returns:
        Safe filename
    """
    # Remove problematic characters
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove control characters
    safe_name = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', safe_name)
    
    # Collapse multiple underscores
    safe_name = re.sub(r'_+', '_', safe_name)
    
    # Trim to max length
    if len(safe_name) > max_length:
        name, ext = os.path.splitext(safe_name)
        available_length = max_length - len(ext)
        safe_name = name[:available_length] + ext
    
    return safe_name.strip('_')


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def parse_duration(duration_str: str) -> float:
    """
    Parse duration string to seconds.
    
    Args:
        duration_str: Duration string like "1h 30m 45s"
        
    Returns:
        Duration in seconds
    """
    total_seconds = 0.0
    
    # Find hours, minutes, seconds
    hours = re.search(r'(\d+)h', duration_str)
    minutes = re.search(r'(\d+)m', duration_str)
    seconds = re.search(r'(\d+(?:\.\d+)?)s', duration_str)
    
    if hours:
        total_seconds += int(hours.group(1)) * 3600
    if minutes:
        total_seconds += int(minutes.group(1)) * 60
    if seconds:
        total_seconds += float(seconds.group(1))
    
    return total_seconds