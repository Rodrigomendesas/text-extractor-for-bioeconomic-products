"""
Test suite for bioeconomic products analysis system.

This package contains comprehensive tests for all components:
- Unit tests for individual modules
- Integration tests for component interactions
- End-to-end tests for complete workflows
- Performance and stress tests

Test Structure:
- test_models.py: Tests for data models and schemas
- test_extractors.py: Tests for extraction implementations
- test_pdf_processor.py: Tests for PDF processing functionality
- test_storage.py: Tests for database and export functionality
- test_llm.py: Tests for LLM integration
- test_utils.py: Tests for utility functions
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any
import logging

# Configure test logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise during tests

# Test configuration
TEST_CONFIG = {
    "database_path": None,  # Will be set to temp path
    "exports_path": None,   # Will be set to temp path
    "logs_path": None,      # Will be set to temp path
    "temp_path": None,      # Will be set to temp path
    "chunk_size": 500,      # Smaller for testing
    "chunk_overlap": 50,
    "confidence_threshold": 0.3,
    "max_retries": 2,
    "log_level": "WARNING"
}


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp(prefix="bioeconomic_test_"))
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture(scope="session")
def test_config(temp_dir: Path) -> Dict[str, Any]:
    """Get test configuration with temporary paths."""
    config = TEST_CONFIG.copy()
    config.update({
        "database_path": temp_dir / "test.db",
        "exports_path": temp_dir / "exports",
        "logs_path": temp_dir / "logs",
        "temp_path": temp_dir / "temp"
    })
    return config


@pytest.fixture
def sample_text_data() -> Dict[str, str]:
    """Sample text data for testing."""
    return {
        "simple_product": """
        Açaí (Euterpe oleracea) is a palm tree native to Brazil, specifically 
        from the Amazon rainforest region. The fruit is traditionally used by 
        indigenous communities for food and medicinal purposes. It has gained 
        commercial popularity as a superfood and is exported worldwide.
        """,
        
        "multiple_products": """
        In the Amazon region of Peru, several important bioeconomic products 
        are harvested sustainably. Camu camu (Myrciaria dubia) contains high 
        levels of vitamin C and is used for nutritional supplements. 
        
        Cat's claw (Uncaria tomentosa) is a medicinal vine used traditionally 
        for immune system support and is now commercialized globally. 
        
        Brazil nut (Bertholletia excelsa) provides protein-rich seeds that 
        are harvested by local communities and sold in international markets.
        """,
        
        "spanish_text": """
        La maca (Lepidium meyenii) es una planta originaria de los Andes 
        peruanos que se cultiva tradicionalmente en las regiones altas. 
        Se utiliza como alimento nutritivo y medicina tradicional para 
        aumentar la energía y la fertilidad. Actualmente se exporta como 
        suplemento dietético.
        """,
        
        "irrelevant_text": """
        The weather forecast for tomorrow shows a 30% chance of rain with 
        temperatures ranging from 15 to 22 degrees Celsius. Traffic 
        conditions on the main highway are expected to be heavy during 
        rush hour. The local soccer team won their match yesterday.
        """,
        
        "mixed_content": """
        Economic development in rural Colombia has been enhanced through 
        sustainable harvesting of natural products. Tagua (Phytelephas aequatorialis), 
        known as vegetable ivory, is carved into buttons and decorative items. 
        
        The weather has been favorable for agriculture this season, with 
        adequate rainfall supporting crop growth. Local farmers also cultivate 
        coffee and plantains for domestic consumption.
        
        Copaiba oil (Copaifera spp.) is extracted from trees in the region 
        and used for medicinal and cosmetic applications.
        """
    }


@pytest.fixture
def sample_product_data() -> Dict[str, Dict[str, Any]]:
    """Sample product data for testing."""
    return {
        "acai": {
            "product_name": "Açaí",
            "scientific_name": "Euterpe oleracea",
            "country": "Brazil",
            "uses": ["Food", "Medicinal", "Commercial superfood"],
            "additional_info": "Palm tree native to Amazon rainforest"
        },
        
        "camu_camu": {
            "product_name": "Camu camu",
            "scientific_name": "Myrciaria dubia", 
            "country": "Peru",
            "uses": ["Nutritional supplement", "High vitamin C content"],
            "additional_info": "Found in Amazon region"
        },
        
        "cats_claw": {
            "product_name": "Cat's claw",
            "scientific_name": "Uncaria tomentosa",
            "country": "Peru", 
            "uses": ["Medicinal", "Immune system support"],
            "additional_info": "Medicinal vine used traditionally"
        }
    }


# Test utilities
def create_sample_pdf(temp_dir: Path, content: str, filename: str = "test.pdf") -> Path:
    """Create a sample PDF file for testing."""
    # This would create an actual PDF in a real implementation
    # For now, create a text file that represents PDF content
    pdf_path = temp_dir / filename
    with open(pdf_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return pdf_path


def assert_product_valid(product, expected_data: Dict[str, Any] = None):
    """Assert that a product object is valid."""
    from src.models import Product
    
    assert isinstance(product, Product)
    assert product.product_name
    assert product.id
    assert product.created_at
    assert product.updated_at
    assert 0.0 <= product.confidence_score <= 1.0
    
    if expected_data:
        assert product.product_name == expected_data.get("product_name")
        if "scientific_name" in expected_data:
            assert product.scientific_name == expected_data["scientific_name"]
        if "country" in expected_data and product.origin:
            assert product.origin.country == expected_data["country"]


def assert_extraction_result_valid(result, min_products: int = 0):
    """Assert that an extraction result is valid."""
    from src.models import ExtractionResult, ProcessingStatus
    
    assert isinstance(result, ExtractionResult)
    assert result.id
    assert result.created_at
    assert isinstance(result.status, ProcessingStatus)
    assert len(result.products) >= min_products
    assert 0.0 <= result.overall_confidence <= 1.0
    
    for product in result.products:
        assert_product_valid(product)


# Test markers
pytest_markers = [
    "unit: marks tests as unit tests",
    "integration: marks tests as integration tests", 
    "slow: marks tests as slow running",
    "requires_api: marks tests that require API access",
    "requires_db: marks tests that require database access"
]

# Register custom markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    for marker in pytest_markers:
        config.addinivalue_line("markers", marker)