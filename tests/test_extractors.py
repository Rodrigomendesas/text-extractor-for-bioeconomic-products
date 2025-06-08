"""Tests for extraction implementations."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.extractors import BaseExtractor, OpenAIExtractor
from src.models import ExtractionResult, Product, ProcessingStatus
from src.llm import OpenAIClient, PromptManager, ResponseParser


class TestBaseExtractor:
    """Test BaseExtractor abstract class."""

    def test_base_extractor_cannot_be_instantiated(self):
        """Test that BaseExtractor cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseExtractor()

    def test_subclass_must_implement_extract(self):
        """Test that subclasses must implement extract method."""
        class IncompleteExtractor(BaseExtractor):
            pass

        with pytest.raises(TypeError):
            IncompleteExtractor()


class MockExtractor(BaseExtractor):
    """Mock extractor for testing."""

    def __init__(self):
        super().__init__()
        self.extract_calls = []

    def extract(self, text: str, **kwargs) -> ExtractionResult:
        """Mock extract method."""
        self.extract_calls.append((text, kwargs))

        # Return a simple result
        result = ExtractionResult()

        if "error" in text.lower():
            result.mark_failed("Mock error")
        else:
            product = Product(product_name="Mock Product")
            result.add_product(product)
            result.mark_completed()

        return result


class TestMockExtractor:
    """Test the mock extractor implementation."""

    def test_mock_extractor_basic_functionality(self):
        """Test basic functionality of mock extractor."""
        extractor = MockExtractor()

        result = extractor.extract("Test text")

        assert len(extractor.extract_calls) == 1
        assert extractor.extract_calls[0][0] == "Test text"
        assert len(result.products) == 1
        assert result.products[0].product_name == "Mock Product"
        assert result.status == ProcessingStatus.COMPLETED

    def test_mock_extractor_error_handling(self):
        """Test error handling in mock extractor."""
        extractor = MockExtractor()

        result = extractor.extract("This text contains error")

        assert result.status == ProcessingStatus.FAILED
        assert len(result.products) == 0


class TestOpenAIExtractor:
    """Test OpenAI-based extractor."""

    @pytest.fixture
    def mock_client(self):
        """Mock OpenAI client."""
        client = Mock(spec=OpenAIClient)
        return client

    @pytest.fixture
    def mock_prompt_manager(self):
        """Mock prompt manager."""
        manager = Mock(spec=PromptManager)
        manager.get_extraction_prompt.return_value = "Mock prompt: {text}"
        return manager

    @pytest.fixture
    def mock_parser(self):
        """Mock response parser."""
        parser = Mock(spec=ResponseParser)
        return parser

    @pytest.fixture
    def extractor(self, mock_client, mock_prompt_manager, mock_parser):
        """Create OpenAI extractor with mocks."""
        return OpenAIExtractor(
            client=mock_client,
            prompt_manager=mock_prompt_manager,
            parser=mock_parser
        )

    def test_extractor_initialization(self, extractor, mock_client, mock_prompt_manager, mock_parser):
        """Test extractor initialization."""
        assert extractor.client == mock_client
        assert extractor.prompt_manager == mock_prompt_manager
        assert extractor.parser == mock_parser
        assert extractor.model == "gpt-3.5-turbo"
        assert extractor.temperature == 0.1
        assert extractor.max_tokens == 2000

    def test_extract_with_successful_response(self, extractor, mock_client, mock_prompt_manager, mock_parser):
        """Test successful extraction."""
        # Setup mocks
        mock_response = {
            "choices": [{
                "message": {
                    "content": '{"products": [{"product_name": "Açaí", "country": "Brazil", "uses": ["Food"]}]}'
                }
            }],
            "usage": {"total_tokens": 500}
        }
        mock_client.get_completion.return_value = mock_response

        mock_parsed_data = {
            "products": [{
                "product_name": "Açaí",
                "country": "Brazil", 
                "uses": ["Food"]
            }]
        }
        mock_parser.parse_extraction_response.return_value = mock_parsed_data

        # Execute extraction
        text = "Açaí is a fruit from Brazil used as food."
        result = extractor.extract(text)

        # Verify calls
        mock_prompt_manager.get_extraction_prompt.assert_called_once()
        mock_client.get_completion.assert_called_once()
        mock_parser.parse_extraction_response.assert_called_once()

        # Verify result
        assert result.status == ProcessingStatus.COMPLETED
        assert len(result.products) == 1
        assert result.products[0].product_name == "Açaí"
        assert result.products[0].origin.country == "Brazil"
        assert len(result.products[0].uses) == 1
        assert result.metadata.tokens_used == 500

    def test_extract_with_api_error(self, extractor, mock_client, mock_prompt_manager, mock_parser):
        """Test extraction with API error."""
        # Setup mock to raise exception
        mock_client.get_completion.side_effect = Exception("API Error")

        # Execute extraction
        result = extractor.extract("Test text")

        # Verify error handling
        assert result.status == ProcessingStatus.FAILED
        assert len(result.products) == 0
        assert result.metadata.errors_encountered
        assert "API Error" in str(result.metadata.errors_encountered)

    def test_extract_with_parsing_error(self, extractor, mock_client, mock_prompt_manager, mock_parser):
        """Test extraction with parsing error."""
        # Setup mocks
        mock_response = {
            "choices": [{"message": {"content": "Invalid JSON"}}],
            "usage": {"total_tokens": 100}
        }
        mock_client.get_completion.return_value = mock_response
        mock_parser.parse_extraction_response.side_effect = ValueError("Parse error")

        # Execute extraction
        result = extractor.extract("Test text")

        # Verify error handling
        assert result.status == ProcessingStatus.FAILED
        assert len(result.products) == 0
        assert result.metadata.errors_encountered

    def test_extract_with_empty_response(self, extractor, mock_client, mock_prompt_manager, mock_parser):
        """Test extraction with empty response."""
        # Setup mocks
        mock_response = {
            "choices": [{"message": {"content": "{}"}}],
            "usage": {"total_tokens": 50}
        }
        mock_client.get_completion.return_value = mock_response
        mock_parser.parse_extraction_response.return_value = {"products": []}

        # Execute extraction
        result = extractor.extract("Irrelevant text about weather")

        # Verify result
        assert result.status == ProcessingStatus.COMPLETED
        assert len(result.products) == 0
        assert result.metadata.tokens_used == 50

    def test_extract_with_multiple_products(self, extractor, mock_client, mock_prompt_manager, mock_parser):
        """Test extraction with multiple products."""
        # Setup mocks
        mock_response = {
            "choices": [{
                "message": {
                    "content": '{"products": []}'  # Content doesn't matter, parser handles it
                }
            }],
            "usage": {"total_tokens": 800}
        }
        mock_client.get_completion.return_value = mock_response

        mock_parsed_data = {
            "products": [
                {
                    "product_name": "Camu camu",
                    "scientific_name": "Myrciaria dubia",
                    "country": "Peru",
                    "uses": ["Nutritional supplement"]
                },
                {
                    "product_name": "Cat's claw", 
                    "scientific_name": "Uncaria tomentosa",
                    "country": "Peru",
                    "uses": ["Medicinal"]
                }
            ]
        }
        mock_parser.parse_extraction_response.return_value = mock_parsed_data

        # Execute extraction
        result = extractor.extract("Text about multiple Peruvian plants")

        # Verify result
        assert result.status == ProcessingStatus.COMPLETED
        assert len(result.products) == 2
        assert result.products[0].product_name == "Camu camu"
        assert result.products[1].product_name == "Cat's claw"
        assert all(p.origin.country == "Peru" for p in result.products)

    def test_extract_with_custom_parameters(self, extractor, mock_client, mock_prompt_manager, mock_parser):
        """Test extraction with custom parameters."""
        # Setup mocks
        mock_response = {
            "choices": [{"message": {"content": "{}"}}],
            "usage": {"total_tokens": 200}
        }
        mock_client.get_completion.return_value = mock_response
        mock_parser.parse_extraction_response.return_value = {"products": []}

        # Execute extraction with custom parameters
        result = extractor.extract(
            "Test text",
            confidence_threshold=0.8,
            source_document="test.pdf",
            custom_prompt="Custom extraction prompt"
        )

        # Verify metadata
        assert result.metadata.confidence_threshold == 0.8
        assert result.metadata.source_file == "test.pdf"

    def test_extract_with_chunked_text(self, extractor, mock_client, mock_prompt_manager, mock_parser):
        """Test extraction with large text that needs chunking."""
        # Setup mocks for multiple chunks
        mock_response = {
            "choices": [{"message": {"content": "{}"}}],
            "usage": {"total_tokens": 300}
        }
        mock_client.get_completion.return_value = mock_response
        mock_parser.parse_extraction_response.return_value = {
            "products": [{
                "product_name": "Test Product",
                "country": "Brazil",
                "uses": ["Test use"]
            }]
        }

        # Create long text that would be chunked
        long_text = "Test content. " * 1000  # Very long text

        # Execute extraction
        result = extractor.extract(long_text, chunk_size=500)

        # Verify that multiple API calls were made for chunks
        assert mock_client.get_completion.call_count > 1
        assert result.status == ProcessingStatus.COMPLETED

    def test_extract_processes_metadata_correctly(self, extractor, mock_client, mock_prompt_manager, mock_parser):
        """Test that extraction metadata is processed correctly."""
        # Setup mocks
        mock_response = {
            "choices": [{"message": {"content": "{}"}}],
            "usage": {"total_tokens": 150}
        }
        mock_client.get_completion.return_value = mock_response
        mock_parser.parse_extraction_response.return_value = {"products": []}

        # Execute extraction
        start_time = pytest.approx(0, abs=1)  # Allow 1 second tolerance
        result = extractor.extract("Test text", source_document="document.pdf")

        # Verify metadata
        assert result.metadata is not None
        assert result.metadata.extraction_method.value == "openai_gpt"
        assert result.metadata.model_used == "gpt-3.5-turbo"
        assert result.metadata.tokens_used == 150
        assert result.metadata.source_file == "document.pdf"
        assert result.metadata.processing_time_seconds is not None
        assert result.metadata.processing_time_seconds > 0

    @pytest.mark.slow
    def test_extract_with_retry_logic(self, extractor, mock_client, mock_prompt_manager, mock_parser):
        """Test extraction with retry logic on temporary failures."""
        # Setup mock to fail twice then succeed
        mock_responses = [
            Exception("Temporary error"),
            Exception("Another temporary error"),
            {
                "choices": [{"message": {"content": "{}"}}],
                "usage": {"total_tokens": 100}
            }
        ]
        mock_client.get_completion.side_effect = mock_responses
        mock_parser.parse_extraction_response.return_value = {"products": []}

        # Execute extraction
        result = extractor.extract("Test text")

        # Verify retries occurred and final success
        assert mock_client.get_completion.call_count == 3
        assert result.status == ProcessingStatus.COMPLETED

    def test_confidence_score_calculation(self, extractor, mock_client, mock_prompt_manager, mock_parser):
        """Test confidence score calculation for extracted products."""
        # Setup mocks
        mock_response = {
            "choices": [{"message": {"content": "{}"}}],
            "usage": {"total_tokens": 200}
        }
        mock_client.get_completion.return_value = mock_response

        mock_parsed_data = {
            "products": [{
                "product_name": "Well-documented product",
                "scientific_name": "Genus species",
                "country": "Brazil",
                "uses": ["Traditional medicine", "Commercial food"],
                "additional_info": "Detailed information about usage and origin"
            }]
        }
        mock_parser.parse_extraction_response.return_value = mock_parsed_data

        # Execute extraction
        result = extractor.extract("Detailed text about a well-known product")

        # Verify confidence score is calculated
        assert len(result.products) == 1
        assert result.products[0].confidence_score > 0
        assert result.overall_confidence > 0

    def test_visual_element_extraction(self, extractor, mock_client, mock_prompt_manager, mock_parser):
        """Test extraction of visual elements from text."""
        from src.models.visual_element import VisualElement, VisualElementType

        # Setup mocks
        mock_response = {
            "choices": [{"message": {"content": "{}"}}],
            "usage": {"total_tokens": 300}
        }
        mock_client.get_completion.return_value = mock_response

        # Mock parsed data with visual elements
        mock_parsed_data = {
            "products": [{
                "product_name": "Product from image",
                "country": "Peru",
                "uses": ["Food"]
            }],
            "visual_elements": [
                {
                    "element_type": "image",
                    "page_number": 1,
                    "description": "Image of a plant",
                    "ocr_text": "Plant species information"
                },
                {
                    "element_type": "table",
                    "page_number": 2,
                    "description": "Table of nutritional values"
                }
            ]
        }
        mock_parser.parse_extraction_response.return_value = mock_parsed_data

        # Create a mock method to handle visual elements
        def mock_process_visual_elements(data, result):
            """Mock method to process visual elements."""
            if "visual_elements" in data:
                for element_data in data["visual_elements"]:
                    element_type = VisualElementType(element_data.get("element_type", "unknown"))
                    element = VisualElement(
                        element_type=element_type,
                        page_number=element_data.get("page_number", 0),
                        description=element_data.get("description", ""),
                        ocr_text=element_data.get("ocr_text", "")
                    )
                    result.add_visual_element(element)

        # Patch the extractor to use our mock method
        with patch.object(extractor, '_process_visual_elements', side_effect=mock_process_visual_elements):
            # Execute extraction
            result = extractor.extract("Text with references to images and tables")

            # Verify visual elements are extracted
            assert len(result.visual_elements) == 2
            assert result.visual_elements[0].element_type == VisualElementType.IMAGE
            assert result.visual_elements[1].element_type == VisualElementType.TABLE
            assert result.visual_elements[0].description == "Image of a plant"
            assert result.visual_elements[0].ocr_text == "Plant species information"
            assert result.visual_elements[1].description == "Table of nutritional values"

            # Verify visual element types are tracked
            assert "image" in result.visual_element_types
            assert "table" in result.visual_element_types
            assert result.visual_element_types["image"] == 1
            assert result.visual_element_types["table"] == 1

            # Verify products are still processed
            assert len(result.products) == 1
            assert result.products[0].product_name == "Product from image"
