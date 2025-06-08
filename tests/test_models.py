"""Tests for data models and schemas."""

import pytest
from datetime import datetime
from typing import List, Dict, Any

from src.models import (
    Product, ProductOrigin, ProductUse, ProductCategory, ProcessingLevel,
    ExtractionResult, ExtractionMetadata, ProcessingStatus, ExtractionMethod,
    VisualElement, VisualElementType
)


class TestProductOrigin:
    """Test ProductOrigin model."""

    def test_create_simple_origin(self):
        """Test creating a simple origin."""
        origin = ProductOrigin(country="Brazil")

        assert origin.country == "Brazil"
        assert origin.region is None
        assert origin.specific_location is None
        assert origin.coordinates is None
        assert origin.ecosystem_type is None

    def test_create_detailed_origin(self):
        """Test creating a detailed origin."""
        origin = ProductOrigin(
            country="Peru",
            region="Amazon",
            specific_location="Iquitos region",
            coordinates={"lat": -3.7437, "lng": -73.2516},
            ecosystem_type="Tropical rainforest"
        )

        assert origin.country == "Peru"
        assert origin.region == "Amazon"
        assert origin.specific_location == "Iquitos region"
        assert origin.coordinates == {"lat": -3.7437, "lng": -73.2516}
        assert origin.ecosystem_type == "Tropical rainforest"

    def test_origin_to_dict(self):
        """Test converting origin to dictionary."""
        origin = ProductOrigin(
            country="Colombia",
            region="Chocó",
            coordinates={"lat": 5.0, "lng": -76.0}
        )

        origin_dict = origin.to_dict()

        assert origin_dict["country"] == "Colombia"
        assert origin_dict["region"] == "Chocó"
        assert origin_dict["coordinates"] == {"lat": 5.0, "lng": -76.0}

    def test_origin_whitespace_cleaning(self):
        """Test that whitespace is cleaned from origin fields."""
        origin = ProductOrigin(
            country="  Brazil  ",
            region="  Amazon  ",
            specific_location="  Manaus area  "
        )

        assert origin.country == "Brazil"
        assert origin.region == "Amazon"
        assert origin.specific_location == "Manaus area"


class TestProductUse:
    """Test ProductUse model."""

    def test_create_simple_use(self):
        """Test creating a simple use."""
        use = ProductUse(
            category=ProductCategory.FOOD,
            description="Consumed as fruit"
        )

        assert use.category == ProductCategory.FOOD
        assert use.description == "Consumed as fruit"
        assert use.traditional_use is False
        assert use.commercial_use is False

    def test_create_detailed_use(self):
        """Test creating a detailed use."""
        use = ProductUse(
            category=ProductCategory.MEDICINE,
            description="Used for immune system support",
            traditional_use=True,
            commercial_use=True,
            market_value="$50-100 per kg",
            sustainability_notes="Sustainably harvested"
        )

        assert use.category == ProductCategory.MEDICINE
        assert use.description == "Used for immune system support"
        assert use.traditional_use is True
        assert use.commercial_use is True
        assert use.market_value == "$50-100 per kg"
        assert use.sustainability_notes == "Sustainably harvested"

    def test_use_to_dict(self):
        """Test converting use to dictionary."""
        use = ProductUse(
            category=ProductCategory.COSMETIC,
            description="Skin care application",
            traditional_use=True
        )

        use_dict = use.to_dict()

        assert use_dict["category"] == "cosmetic"
        assert use_dict["description"] == "Skin care application"
        assert use_dict["traditional_use"] is True


class TestProduct:
    """Test Product model."""

    def test_create_minimal_product(self):
        """Test creating a product with minimal data."""
        product = Product(product_name="Test Product")

        assert product.product_name == "Test Product"
        assert product.scientific_name is None
        assert product.common_names == []
        assert product.origin is None
        assert product.uses == []
        assert product.processing_level == ProcessingLevel.RAW
        assert product.additional_info == ""
        assert product.id is not None
        assert isinstance(product.created_at, datetime)
        assert isinstance(product.updated_at, datetime)
        assert product.confidence_score == 0.0

    def test_create_complete_product(self):
        """Test creating a product with complete data."""
        origin = ProductOrigin(country="Brazil", region="Amazon")
        use = ProductUse(category=ProductCategory.FOOD, description="Nutritious fruit")

        product = Product(
            product_name="Açaí",
            scientific_name="Euterpe oleracea",
            common_names=["Açaí palm", "Cabbage palm"],
            origin=origin,
            uses=[use],
            processing_level=ProcessingLevel.MINIMALLY_PROCESSED,
            additional_info="Superfood from Amazon",
            confidence_score=0.9
        )

        assert product.product_name == "Açaí"
        assert product.scientific_name == "Euterpe oleracea"
        assert product.common_names == ["Açaí palm", "Cabbage palm"]
        assert product.origin == origin
        assert product.uses == [use]
        assert product.processing_level == ProcessingLevel.MINIMALLY_PROCESSED
        assert product.additional_info == "Superfood from Amazon"
        assert product.confidence_score == 0.9

    def test_product_name_required(self):
        """Test that product name is required."""
        with pytest.raises(ValueError, match="Product name is required"):
            Product(product_name="")

    def test_scientific_name_validation(self):
        """Test scientific name validation."""
        # Valid scientific name
        product = Product(
            product_name="Test",
            scientific_name="Genus species"
        )
        assert product.scientific_name == "Genus species"

        # Should capitalize genus and lowercase species
        product = Product(
            product_name="Test",
            scientific_name="genus SPECIES"
        )
        assert product.scientific_name == "Genus species"

    def test_add_use(self):
        """Test adding a use to product."""
        product = Product(product_name="Test Product")

        product.add_use(
            category=ProductCategory.MEDICINE,
            description="Medicinal use",
            traditional_use=True
        )

        assert len(product.uses) == 1
        assert product.uses[0].category == ProductCategory.MEDICINE
        assert product.uses[0].description == "Medicinal use"
        assert product.uses[0].traditional_use is True

    def test_add_common_name(self):
        """Test adding common names."""
        product = Product(product_name="Test Product")

        product.add_common_name("Common Name 1")
        product.add_common_name("Common Name 2")
        product.add_common_name("Common Name 1")  # Duplicate

        assert len(product.common_names) == 2
        assert "Common Name 1" in product.common_names
        assert "Common Name 2" in product.common_names

    def test_set_origin(self):
        """Test setting product origin."""
        product = Product(product_name="Test Product")

        product.set_origin(country="Peru", region="Amazon")

        assert product.origin is not None
        assert product.origin.country == "Peru"
        assert product.origin.region == "Amazon"

    def test_get_use_categories(self):
        """Test getting use categories."""
        product = Product(product_name="Test Product")

        product.add_use(ProductCategory.FOOD, "Food use")
        product.add_use(ProductCategory.MEDICINE, "Medicine use")
        product.add_use(ProductCategory.FOOD, "Another food use")

        categories = product.get_use_categories()
        assert len(categories) == 2
        assert "food" in categories
        assert "medicine" in categories

    def test_is_traditional_product(self):
        """Test checking if product has traditional uses."""
        product = Product(product_name="Test Product")
        assert not product.is_traditional_product()

        product.add_use(ProductCategory.FOOD, "Food use", traditional_use=True)
        assert product.is_traditional_product()

    def test_is_commercial_product(self):
        """Test checking if product has commercial uses."""
        product = Product(product_name="Test Product")
        assert not product.is_commercial_product()

        product.add_use(ProductCategory.FOOD, "Food use", commercial_use=True)
        assert product.is_commercial_product()

    def test_product_to_dict(self):
        """Test converting product to dictionary."""
        origin = ProductOrigin(country="Brazil")
        use = ProductUse(category=ProductCategory.FOOD, description="Food use")

        product = Product(
            product_name="Test Product",
            scientific_name="Test species",
            origin=origin,
            uses=[use]
        )

        product_dict = product.to_dict()

        assert product_dict["product_name"] == "Test Product"
        assert product_dict["scientific_name"] == "Test species"
        assert product_dict["origin"]["country"] == "Brazil"
        assert len(product_dict["uses"]) == 1
        assert product_dict["uses"][0]["category"] == "food"

    def test_product_from_dict(self):
        """Test creating product from dictionary."""
        data = {
            "product_name": "Test Product",
            "scientific_name": "Test species",
            "common_names": ["Name 1", "Name 2"],
            "origin": {
                "country": "Colombia",
                "region": "Pacific"
            },
            "uses": [{
                "category": "medicine",
                "description": "Medicinal use",
                "traditional_use": True,
                "commercial_use": False
            }],
            "processing_level": "processed",
            "additional_info": "Test info",
            "confidence_score": 0.8
        }

        product = Product.from_dict(data)

        assert product.product_name == "Test Product"
        assert product.scientific_name == "Test species"
        assert product.common_names == ["Name 1", "Name 2"]
        assert product.origin.country == "Colombia"
        assert product.origin.region == "Pacific"
        assert len(product.uses) == 1
        assert product.uses[0].category == ProductCategory.MEDICINE
        assert product.uses[0].traditional_use is True
        assert product.processing_level == ProcessingLevel.PROCESSED
        assert product.confidence_score == 0.8

    def test_product_from_extraction_dict(self):
        """Test creating product from simple extraction dictionary."""
        data = {
            "product_name": "Camu camu",
            "scientific_name": "Myrciaria dubia",
            "country": "Peru",
            "uses": ["High vitamin C", "Nutritional supplement"],
            "additional_info": "Amazon superfruit"
        }

        product = Product.from_extraction_dict(
            data,
            confidence_score=0.7,
            source_document="test.pdf"
        )

        assert product.product_name == "Camu camu"
        assert product.scientific_name == "Myrciaria dubia"
        assert product.origin.country == "Peru"
        assert len(product.uses) == 2
        assert product.confidence_score == 0.7
        assert product.source_document == "test.pdf"

    def test_categorize_use(self):
        """Test automatic use categorization."""
        # Test food categorization
        category = Product._categorize_use("Used as food and nutrition")
        assert category == ProductCategory.FOOD

        # Test medicine categorization
        category = Product._categorize_use("Traditional medicinal treatment")
        assert category == ProductCategory.MEDICINE

        # Test cosmetic categorization
        category = Product._categorize_use("Skin care and beauty")
        assert category == ProductCategory.COSMETIC

        # Test default categorization
        category = Product._categorize_use("Unknown usage")
        assert category == ProductCategory.OTHER


class TestExtractionMetadata:
    """Test ExtractionMetadata model."""

    def test_create_metadata(self):
        """Test creating extraction metadata."""
        metadata = ExtractionMetadata(
            extraction_method=ExtractionMethod.OPENAI_GPT,
            model_used="gpt-3.5-turbo",
            tokens_used=1500,
            processing_time_seconds=12.5
        )

        assert metadata.extraction_method == ExtractionMethod.OPENAI_GPT
        assert metadata.model_used == "gpt-3.5-turbo"
        assert metadata.tokens_used == 1500
        assert metadata.processing_time_seconds == 12.5
        assert metadata.confidence_threshold == 0.5
        assert metadata.validation_applied is False

    def test_add_error_and_warning(self):
        """Test adding errors and warnings."""
        metadata = ExtractionMetadata(extraction_method=ExtractionMethod.MANUAL)

        metadata.add_error("Test error")
        metadata.add_warning("Test warning")

        assert len(metadata.errors_encountered) == 1
        assert len(metadata.warnings) == 1
        assert "Test error" in metadata.errors_encountered
        assert "Test warning" in metadata.warnings

    def test_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        metadata = ExtractionMetadata(
            extraction_method=ExtractionMethod.HYBRID,
            tokens_used=500
        )
        metadata.add_error("Test error")

        metadata_dict = metadata.to_dict()

        assert metadata_dict["extraction_method"] == "hybrid"
        assert metadata_dict["tokens_used"] == 500
        assert "Test error" in metadata_dict["errors_encountered"]


class TestExtractionResult:
    """Test ExtractionResult model."""

    def test_create_empty_result(self):
        """Test creating an empty extraction result."""
        result = ExtractionResult()

        assert result.products == []
        assert result.status == ProcessingStatus.PENDING
        assert result.overall_confidence == 0.0
        assert result.total_products_found == 0
        assert result.unique_countries == []
        assert result.product_categories == {}
        assert result.id is not None
        assert isinstance(result.created_at, datetime)

    def test_add_product(self):
        """Test adding a product to result."""
        result = ExtractionResult()
        product = Product(product_name="Test Product")
        product.set_origin(country="Brazil")
        product.add_use(ProductCategory.FOOD, "Food use")

        result.add_product(product)

        assert len(result.products) == 1
        assert result.total_products_found == 1
        assert "Brazil" in result.unique_countries
        assert "food" in result.product_categories
        assert result.product_categories["food"] == 1

    def test_add_multiple_products(self):
        """Test adding multiple products."""
        result = ExtractionResult()

        product1 = Product(product_name="Product 1")
        product1.set_origin(country="Brazil")
        product1.add_use(ProductCategory.FOOD, "Food use")

        product2 = Product(product_name="Product 2")
        product2.set_origin(country="Peru")
        product2.add_use(ProductCategory.MEDICINE, "Medicine use")

        result.add_products([product1, product2])

        assert len(result.products) == 2
        assert result.total_products_found == 2
        assert set(result.unique_countries) == {"Brazil", "Peru"}
        assert result.product_categories == {"food": 1, "medicine": 1}

    def test_mark_completed(self):
        """Test marking result as completed."""
        result = ExtractionResult()

        result.mark_completed()

        assert result.status == ProcessingStatus.COMPLETED
        assert result.completed_at is not None

    def test_mark_failed(self):
        """Test marking result as failed."""
        metadata = ExtractionMetadata(extraction_method=ExtractionMethod.MANUAL)
        result = ExtractionResult(metadata=metadata)

        result.mark_failed("Test error")

        assert result.status == ProcessingStatus.FAILED
        assert result.completed_at is not None
        assert "Test error" in metadata.errors_encountered

    def test_filter_by_confidence(self):
        """Test filtering products by confidence."""
        result = ExtractionResult()

        product1 = Product(product_name="High Confidence", confidence_score=0.9)
        product2 = Product(product_name="Low Confidence", confidence_score=0.3)

        result.add_products([product1, product2])

        filtered_result = result.filter_by_confidence(0.5)

        assert len(filtered_result.products) == 1
        assert filtered_result.products[0].product_name == "High Confidence"

    def test_filter_by_country(self):
        """Test filtering products by country."""
        result = ExtractionResult()

        product1 = Product(product_name="Brazil Product")
        product1.set_origin(country="Brazil")

        product2 = Product(product_name="Peru Product")
        product2.set_origin(country="Peru")

        result.add_products([product1, product2])

        filtered_result = result.filter_by_country(["Brazil"])

        assert len(filtered_result.products) == 1
        assert filtered_result.products[0].product_name == "Brazil Product"

    def test_get_products_by_category(self):
        """Test getting products by category."""
        result = ExtractionResult()

        product1 = Product(product_name="Food Product")
        product1.add_use(ProductCategory.FOOD, "Food use")

        product2 = Product(product_name="Medicine Product")
        product2.add_use(ProductCategory.MEDICINE, "Medicine use")

        result.add_products([product1, product2])

        food_products = result.get_products_by_category("food")

        assert len(food_products) == 1
        assert food_products[0].product_name == "Food Product"

    def test_get_summary(self):
        """Test getting result summary."""
        metadata = ExtractionMetadata(
            extraction_method=ExtractionMethod.OPENAI_GPT,
            processing_time_seconds=10.0
        )
        result = ExtractionResult(metadata=metadata)

        product = Product(product_name="Test Product", confidence_score=0.8)
        product.set_origin(country="Colombia")
        product.add_use(ProductCategory.COSMETIC, "Cosmetic use")

        result.add_product(product)
        result.mark_completed()

        summary = result.get_summary()

        assert summary["total_products"] == 1
        assert summary["unique_countries"] == 1
        assert "Colombia" in summary["countries"]
        assert summary["product_categories"]["cosmetic"] == 1
        assert summary["overall_confidence"] == 0.8
        assert summary["processing_time_seconds"] == 10.0
        assert summary["status"] == "completed"

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = ExtractionResult()
        product = Product(product_name="Test Product")
        result.add_product(product)

        result_dict = result.to_dict()

        assert result_dict["id"] == result.id
        assert len(result_dict["products"]) == 1
        assert result_dict["total_products_found"] == 1
        assert result_dict["status"] == "pending"

    def test_result_from_dict(self):
        """Test creating result from dictionary."""
        data = {
            "status": "completed",
            "overall_confidence": 0.7,
            "products": [{
                "product_name": "Test Product",
                "scientific_name": "Test species",
                "origin": {"country": "Venezuela"},
                "uses": [{
                    "category": "industrial",
                    "description": "Industrial use",
                    "traditional_use": False,
                    "commercial_use": True
                }],
                "processing_level": "raw",
                "additional_info": "",
                "confidence_score": 0.7
            }],
            "total_products_found": 1,
            "unique_countries": ["Venezuela"],
            "product_categories": {"industrial": 1}
        }

        result = ExtractionResult.from_dict(data)

        assert result.status == ProcessingStatus.COMPLETED
        assert result.overall_confidence == 0.7
        assert len(result.products) == 1
        assert result.products[0].product_name == "Test Product"
        assert result.total_products_found == 1
        assert "Venezuela" in result.unique_countries


class TestVisualElement:
    """Test VisualElement model."""

    def test_create_minimal_visual_element(self):
        """Test creating a visual element with minimal data."""
        element = VisualElement(element_type=VisualElementType.IMAGE)

        assert element.element_type == VisualElementType.IMAGE
        assert element.page_number == 0
        assert element.position == (0, 0, 0, 0)
        assert element.content_base64 is None
        assert element.content_format == "png"
        assert element.width == 0
        assert element.height == 0
        assert element.dpi == 0
        assert element.id is not None
        assert isinstance(element.extracted_at, datetime)

    def test_create_complete_visual_element(self):
        """Test creating a visual element with complete data."""
        element = VisualElement(
            element_type=VisualElementType.TABLE,
            page_number=5,
            position=(100, 200, 300, 400),
            content_base64="base64_encoded_content",
            content_format="jpg",
            width=200,
            height=100,
            dpi=300,
            caption="Sample table",
            description="A table showing data",
            ocr_text="Table content",
            confidence_score=0.85,
            tags=["table", "data"]
        )

        assert element.element_type == VisualElementType.TABLE
        assert element.page_number == 5
        assert element.position == (100, 200, 300, 400)
        assert element.content_base64 == "base64_encoded_content"
        assert element.content_format == "jpg"
        assert element.width == 200
        assert element.height == 100
        assert element.dpi == 300
        assert element.caption == "Sample table"
        assert element.description == "A table showing data"
        assert element.ocr_text == "Table content"
        assert element.confidence_score == 0.85
        assert element.tags == ["table", "data"]

    def test_visual_element_to_dict(self):
        """Test converting visual element to dictionary."""
        element = VisualElement(
            element_type=VisualElementType.CHART,
            page_number=3,
            position=(50, 60, 250, 180),
            width=200,
            height=120
        )

        element_dict = element.to_dict()

        assert element_dict["element_type"] == "chart"
        assert element_dict["page_number"] == 3
        assert element_dict["position"] == (50, 60, 250, 180)
        assert element_dict["width"] == 200
        assert element_dict["height"] == 120
        assert "id" in element_dict
        assert "extracted_at" in element_dict

    def test_visual_element_from_dict(self):
        """Test creating visual element from dictionary."""
        data = {
            "id": "test-id-123",
            "element_type": "image",
            "page_number": 2,
            "position": (10, 20, 110, 120),
            "content_base64": "test_base64_content",
            "content_format": "png",
            "width": 100,
            "height": 100,
            "dpi": 72,
            "caption": "Test image",
            "description": "An image for testing",
            "confidence_score": 0.75,
            "tags": ["test", "image"]
        }

        element = VisualElement.from_dict(data)

        assert element.id == "test-id-123"
        assert element.element_type == VisualElementType.IMAGE
        assert element.page_number == 2
        assert element.position == (10, 20, 110, 120)
        assert element.content_base64 == "test_base64_content"
        assert element.content_format == "png"
        assert element.width == 100
        assert element.height == 100
        assert element.caption == "Test image"
        assert element.description == "An image for testing"
        assert element.confidence_score == 0.75
        assert element.tags == ["test", "image"]

    def test_visual_element_str_representation(self):
        """Test string representation of visual element."""
        element = VisualElement(
            element_type=VisualElementType.IMAGE,
            page_number=1
        )

        str_repr = str(element)

        assert "VisualElement" in str_repr
        assert "image" in str_repr
        assert "page=1" in str_repr
