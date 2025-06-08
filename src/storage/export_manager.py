"""Export functionality for bioeconomic products data."""

import logging
import json
import csv
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from enum import Enum
import pandas as pd

from src.models import Product, ExtractionResult
from src.storage.database import ProductDatabase

logger = logging.getLogger(__name__)


class ExportFormat(Enum):
    """Available export formats."""
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"
    XML = "xml"
    HTML = "html"


class ExportManager:
    """Manages data export in various formats."""

    def __init__(self, db: Optional[ProductDatabase] = None):
        """
        Initialize export manager.

        Args:
            db: ProductDatabase instance
        """
        self.db = db or ProductDatabase()
        self.logger = logging.getLogger(__name__)

    def export_products(
            self,
            products: List[Product],
            output_path: Path,
            format: ExportFormat,
            include_metadata: bool = True
    ) -> bool:
        """
        Export products to specified format.

        Args:
            products: List of products to export
            output_path: Output file path
            format: Export format
            include_metadata: Whether to include metadata fields

        Returns:
            True if export successful
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if format == ExportFormat.JSON:
                return self._export_json(products, output_path, include_metadata)
            elif format == ExportFormat.CSV:
                return self._export_csv(products, output_path, include_metadata)
            elif format == ExportFormat.EXCEL:
                return self._export_excel(products, output_path, include_metadata)
            elif format == ExportFormat.XML:
                return self._export_xml(products, output_path, include_metadata)
            elif format == ExportFormat.HTML:
                return self._export_html(products, output_path, include_metadata)
            else:
                raise ValueError(f"Unsupported export format: {format}")

        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            return False

    def export_extraction_result(
            self,
            result: ExtractionResult,
            output_path: Path,
            format: ExportFormat,
            include_metadata: bool = True
    ) -> bool:
        """
        Export extraction result to specified format.

        Args:
            result: ExtractionResult to export
            output_path: Output file path
            format: Export format
            include_metadata: Whether to include metadata fields

        Returns:
            True if export successful
        """
        try:
            # Create a comprehensive export including result metadata
            if format == ExportFormat.JSON:
                return self._export_extraction_json(result, output_path, include_metadata)
            else:
                # For other formats, export just the products
                return self.export_products(result.products, output_path, format, include_metadata)

        except Exception as e:
            self.logger.error(f"Extraction result export failed: {e}")
            return False

    def _export_json(self, products: List[Product], output_path: Path, include_metadata: bool) -> bool:
        """Export products to JSON format."""
        try:
            data = {
                "export_info": {
                    "timestamp": datetime.now().isoformat(),
                    "total_products": len(products),
                    "format": "bioeconomic_products_v1"
                },
                "products": [self._product_to_dict(product, include_metadata) for product in products]
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Exported {len(products)} products to JSON: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"JSON export failed: {e}")
            return False

    def _export_csv(self, products: List[Product], output_path: Path, include_metadata: bool) -> bool:
        """Export products to CSV format."""
        try:
            # Flatten product data for CSV
            rows = []
            for product in products:
                base_row = {
                    'product_name': product.product_name,
                    'scientific_name': product.scientific_name or '',
                    'common_names': '; '.join(product.common_names),
                    'country': product.origin.country if product.origin else '',
                    'region': product.origin.region if product.origin else '',
                    'processing_level': product.processing_level.value,
                    'additional_info': product.additional_info
                }

                if include_metadata:
                    base_row.update({
                        'id': product.id,
                        'confidence_score': product.confidence_score,
                        'source_document': product.source_document or '',
                        'extraction_method': product.extraction_method or '',
                        'created_at': product.created_at.isoformat(),
                        'updated_at': product.updated_at.isoformat()
                    })

                # Handle multiple uses
                if product.uses:
                    for i, use in enumerate(product.uses):
                        row = base_row.copy()
                        row.update({
                            'use_category': use.category.value,
                            'use_description': use.description,
                            'traditional_use': use.traditional_use,
                            'commercial_use': use.commercial_use,
                            'market_value': use.market_value or '',
                            'sustainability_notes': use.sustainability_notes or ''
                        })
                        rows.append(row)
                else:
                    # Product with no uses
                    base_row.update({
                        'use_category': '',
                        'use_description': '',
                        'traditional_use': False,
                        'commercial_use': False,
                        'market_value': '',
                        'sustainability_notes': ''
                    })
                    rows.append(base_row)

            # Write CSV
            if rows:
                with open(output_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)

            self.logger.info(f"Exported {len(products)} products to CSV: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"CSV export failed: {e}")
            return False

    def _export_excel(self, products: List[Product], output_path: Path, include_metadata: bool) -> bool:
        """Export products to Excel format."""
        try:
            # Create DataFrames for different sheets
            products_data = []
            uses_data = []

            for product in products:
                # Product sheet data
                product_row = {
                    'ID': product.id if include_metadata else None,
                    'Product Name': product.product_name,
                    'Scientific Name': product.scientific_name or '',
                    'Common Names': '; '.join(product.common_names),
                    'Country': product.origin.country if product.origin else '',
                    'Region': product.origin.region if product.origin else '',
                    'Specific Location': product.origin.specific_location if product.origin else '',
                    'Ecosystem Type': product.origin.ecosystem_type if product.origin else '',
                    'Processing Level': product.processing_level.value,
                    'Additional Info': product.additional_info,
                    'Number of Uses': len(product.uses)
                }

                if include_metadata:
                    product_row.update({
                        'Confidence Score': product.confidence_score,
                        'Source Document': product.source_document or '',
                        'Extraction Method': product.extraction_method or '',
                        'Created At': product.created_at.isoformat(),
                        'Updated At': product.updated_at.isoformat()
                    })

                products_data.append(product_row)

                # Uses sheet data
                for use in product.uses:
                    use_row = {
                        'Product ID': product.id if include_metadata else product.product_name,
                        'Product Name': product.product_name,
                        'Use Category': use.category.value,
                        'Use Description': use.description,
                        'Traditional Use': use.traditional_use,
                        'Commercial Use': use.commercial_use,
                        'Market Value': use.market_value or '',
                        'Sustainability Notes': use.sustainability_notes or ''
                    }
                    uses_data.append(use_row)

            # Create Excel file with multiple sheets
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Products sheet
                if products_data:
                    df_products = pd.DataFrame(products_data)
                    df_products.to_excel(writer, sheet_name='Products', index=False)

                # Uses sheet
                if uses_data:
                    df_uses = pd.DataFrame(uses_data)
                    df_uses.to_excel(writer, sheet_name='Product Uses', index=False)

                # Summary sheet
                summary_data = self._create_summary_data(products)
                df_summary = pd.DataFrame(summary_data)
                df_summary.to_excel(writer, sheet_name='Summary', index=False)

            self.logger.info(f"Exported {len(products)} products to Excel: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Excel export failed: {e}")
            return False

    def _export_xml(self, products: List[Product], output_path: Path, include_metadata: bool) -> bool:
        """Export products to XML format."""
        try:
            root = ET.Element("bioeconomic_products")

            # Add export info
            export_info = ET.SubElement(root, "export_info")
            ET.SubElement(export_info, "timestamp").text = datetime.now().isoformat()
            ET.SubElement(export_info, "total_products").text = str(len(products))
            ET.SubElement(export_info, "format").text = "bioeconomic_products_v1"

            # Add products
            products_elem = ET.SubElement(root, "products")

            for product in products:
                product_elem = ET.SubElement(products_elem, "product")

                if include_metadata:
                    product_elem.set("id", product.id)

                # Basic fields
                ET.SubElement(product_elem, "product_name").text = product.product_name
                if product.scientific_name:
                    ET.SubElement(product_elem, "scientific_name").text = product.scientific_name

                # Common names
                if product.common_names:
                    common_names_elem = ET.SubElement(product_elem, "common_names")
                    for name in product.common_names:
                        ET.SubElement(common_names_elem, "name").text = name

                # Origin
                if product.origin:
                    origin_elem = ET.SubElement(product_elem, "origin")
                    ET.SubElement(origin_elem, "country").text = product.origin.country
                    if product.origin.region:
                        ET.SubElement(origin_elem, "region").text = product.origin.region
                    if product.origin.specific_location:
                        ET.SubElement(origin_elem, "location").text = product.origin.specific_location
                    if product.origin.ecosystem_type:
                        ET.SubElement(origin_elem, "ecosystem").text = product.origin.ecosystem_type

                # Uses
                if product.uses:
                    uses_elem = ET.SubElement(product_elem, "uses")
                    for use in product.uses:
                        use_elem = ET.SubElement(uses_elem, "use")
                        use_elem.set("category", use.category.value)
                        ET.SubElement(use_elem, "description").text = use.description
                        ET.SubElement(use_elem, "traditional").text = str(use.traditional_use).lower()
                        ET.SubElement(use_elem, "commercial").text = str(use.commercial_use).lower()
                        if use.market_value:
                            ET.SubElement(use_elem, "market_value").text = use.market_value

                # Additional info
                ET.SubElement(product_elem, "processing_level").text = product.processing_level.value
                if product.additional_info:
                    ET.SubElement(product_elem, "additional_info").text = product.additional_info

                # Metadata
                if include_metadata:
                    metadata_elem = ET.SubElement(product_elem, "metadata")
                    ET.SubElement(metadata_elem, "confidence_score").text = str(product.confidence_score)
                    if product.source_document:
                        ET.SubElement(metadata_elem, "source_document").text = product.source_document
                    if product.extraction_method:
                        ET.SubElement(metadata_elem, "extraction_method").text = product.extraction_method
                    ET.SubElement(metadata_elem, "created_at").text = product.created_at.isoformat()
                    ET.SubElement(metadata_elem, "updated_at").text = product.updated_at.isoformat()

            # Write XML
            tree = ET.ElementTree(root)
            tree.write(output_path, encoding='utf-8', xml_declaration=True)

            self.logger.info(f"Exported {len(products)} products to XML: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"XML export failed: {e}")
            return False

    def _export_html(self, products: List[Product], output_path: Path, include_metadata: bool) -> bool:
        """Export products to HTML format."""
        try:
            html_content = self._generate_html_report(products, include_metadata)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            self.logger.info(f"Exported {len(products)} products to HTML: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"HTML export failed: {e}")
            return False

    def _export_extraction_json(self, result: ExtractionResult, output_path: Path, include_metadata: bool) -> bool:
        """Export complete extraction result to JSON."""
        try:
            data = {
                "extraction_result": {
                    "id": result.id,
                    "status": result.status.value,
                    "overall_confidence": result.overall_confidence,
                    "total_products_found": result.total_products_found,
                    "unique_countries": result.unique_countries,
                    "product_categories": result.product_categories,
                    "created_at": result.created_at.isoformat(),
                    "completed_at": result.completed_at.isoformat() if result.completed_at else None
                },
                "metadata": result.metadata.to_dict() if result.metadata else None,
                "products": [self._product_to_dict(product, include_metadata) for product in result.products]
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Exported extraction result {result.id} to JSON: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Extraction JSON export failed: {e}")
            return False

    def _product_to_dict(self, product: Product, include_metadata: bool) -> Dict[str, Any]:
        """Convert product to dictionary for export."""
        data = {
            "product_name": product.product_name,
            "scientific_name": product.scientific_name,
            "common_names": product.common_names,
            "origin": product.origin.to_dict() if product.origin else None,
            "uses": [use.to_dict() for use in product.uses],
            "processing_level": product.processing_level.value,
            "additional_info": product.additional_info
        }

        if include_metadata:
            data.update({
                "id": product.id,
                "confidence_score": product.confidence_score,
                "source_document": product.source_document,
                "extraction_method": product.extraction_method,
                "created_at": product.created_at.isoformat(),
                "updated_at": product.updated_at.isoformat()
            })

        return data

    def _create_summary_data(self, products: List[Product]) -> List[Dict[str, Any]]:
        """Create summary statistics for export."""
        # Country distribution
        countries = {}
        categories = {}

        for product in products:
            if product.origin:
                country = product.origin.country
                countries[country] = countries.get(country, 0) + 1

            for use in product.uses:
                category = use.category.value
                categories[category] = categories.get(category, 0) + 1

        summary = [
            {"Metric": "Total Products", "Value": len(products)},
            {"Metric": "Total Countries", "Value": len(countries)},
            {"Metric": "Total Use Categories", "Value": len(categories)},
            {"Metric": "Average Confidence Score",
             "Value": sum(p.confidence_score for p in products) / len(products) if products else 0}
        ]

        # Add country breakdown
        summary.append({"Metric": "--- Country Distribution ---", "Value": ""})
        for country, count in sorted(countries.items(), key=lambda x: x[1], reverse=True):
            summary.append({"Metric": f"  {country}", "Value": count})

        # Add category breakdown
        summary.append({"Metric": "--- Category Distribution ---", "Value": ""})
        for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            summary.append({"Metric": f"  {category}", "Value": count})

        return summary

    def _generate_html_report(self, products: List[Product], include_metadata: bool) -> str:
        """Generate HTML report for products."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Bioeconomic Products Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .product {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }}
                .product-name {{ font-size: 18px; font-weight: bold; color: #2c5f41; }}
                .scientific-name {{ font-style: italic; color: #666; }}
                .country {{ background-color: #e8f5e8; padding: 5px; border-radius: 3px; display: inline-block; }}
                .use {{ background-color: #f9f9f9; margin: 5px 0; padding: 8px; border-left: 3px solid #2c5f41; }}
                .metadata {{ font-size: 12px; color: #888; margin-top: 10px; }}
                .summary {{ background-color: #f8f9fa; padding: 15px; margin: 20px 0; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Bioeconomic Products Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Total Products: {len(products)}</p>
            </div>
        """

        # Add summary
        countries = set()
        categories = set()
        for product in products:
            if product.origin:
                countries.add(product.origin.country)
            for use in product.uses:
                categories.add(use.category.value)

        html += f"""
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Countries:</strong> {len(countries)} ({', '.join(sorted(countries))})</p>
                <p><strong>Categories:</strong> {len(categories)} ({', '.join(sorted(categories))})</p>
            </div>
        """

        # Add products
        for product in products:
            html += f"""
                <div class="product">
                    <div class="product-name">{product.product_name}</div>
            """

            if product.scientific_name:
                html += f'<div class="scientific-name">{product.scientific_name}</div>'

            if product.origin:
                html += f'<div class="country">{product.origin.country}</div>'

            if product.uses:
                html += '<h4>Uses:</h4>'
                for use in product.uses:
                    html += f"""
                        <div class="use">
                            <strong>{use.category.value.title()}:</strong> {use.description}
                            {'(Traditional)' if use.traditional_use else ''}
                            {'(Commercial)' if use.commercial_use else ''}
                        </div>
                    """

            if product.additional_info:
                html += f'<p><strong>Additional Info:</strong> {product.additional_info}</p>'

            if include_metadata:
                html += f"""
                    <div class="metadata">
                        Confidence: {product.confidence_score:.3f} | 
                        Created: {product.created_at.strftime('%Y-%m-%d')}
                        {f"| Source: {product.source_document}" if product.source_document else ""}
                    </div>
                """

            html += '</div>'

        html += """
        </body>
        </html>
        """

        return html