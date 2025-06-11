"""Database management for bioeconomic products storage."""

import logging
import sqlite3
import json
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from contextlib import contextmanager

from src.models import Product, ExtractionResult, ProcessingStatus, ExtractionMethod
from config import settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages SQLite database connections and operations."""

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize database manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path or (settings.database_dir / "products.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()
        logger.info(f"Database initialized at: {self.db_path}")

    def _init_database(self):
        """Initialize database schema."""
        with self.get_connection() as conn:
            self._create_tables(conn)

    def _create_tables(self, conn: sqlite3.Connection):
        """Create database tables."""

        # Products table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id TEXT PRIMARY KEY,
                product_name TEXT NOT NULL,
                scientific_name TEXT,
                common_names TEXT,  -- JSON array
                origin_country TEXT,
                origin_region TEXT,
                origin_location TEXT,
                origin_coordinates TEXT,  -- JSON object
                origin_ecosystem TEXT,
                processing_level TEXT,
                additional_info TEXT,
                confidence_score REAL,
                source_document TEXT,
                extraction_method TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)

        # Product uses table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS product_uses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id TEXT,
                category TEXT,
                description TEXT,
                traditional_use BOOLEAN,
                commercial_use BOOLEAN,
                market_value TEXT,
                sustainability_notes TEXT,
                FOREIGN KEY (product_id) REFERENCES products (id)
            )
        """)

        # Extraction results table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS extraction_results (
                id TEXT PRIMARY KEY,
                status TEXT,
                overall_confidence REAL,
                total_products_found INTEGER,
                unique_countries TEXT,  -- JSON array
                product_categories TEXT,  -- JSON object
                extraction_method TEXT,
                model_used TEXT,
                tokens_used INTEGER,
                processing_time_seconds REAL,
                source_file TEXT,
                source_file_size INTEGER,
                source_pages INTEGER,
                text_length INTEGER,
                chunks_processed INTEGER,
                total_chunks INTEGER,
                errors_encountered TEXT,  -- JSON array
                warnings TEXT,  -- JSON array
                validation_applied BOOLEAN,
                confidence_threshold REAL,
                created_at TEXT,
                completed_at TEXT
            )
        """)

        # Link table for extraction results and products
        conn.execute("""
            CREATE TABLE IF NOT EXISTS extraction_products (
                extraction_id TEXT,
                product_id TEXT,
                PRIMARY KEY (extraction_id, product_id),
                FOREIGN KEY (extraction_id) REFERENCES extraction_results (id),
                FOREIGN KEY (product_id) REFERENCES products (id)
            )
        """)

        # Create indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_products_name ON products (product_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_products_country ON products (origin_country)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_products_confidence ON products (confidence_score)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_uses_category ON product_uses (category)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_extraction_status ON extraction_results (status)")

        conn.commit()

    @contextmanager
    def get_connection(self):
        """Get database connection with context manager."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()

    def backup_database(self, backup_path: Path):
        """
        Create a backup of the database.

        Args:
            backup_path: Path for backup file
        """
        with self.get_connection() as source:
            backup_conn = sqlite3.connect(str(backup_path))
            source.backup(backup_conn)
            backup_conn.close()
        logger.info(f"Database backed up to: {backup_path}")

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self.get_connection() as conn:
            stats = {}

            # Count tables
            tables = ['products', 'product_uses', 'extraction_results', 'extraction_products']
            for table in tables:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f"{table}_count"] = cursor.fetchone()[0]

            # Database file size
            stats['database_size_bytes'] = self.db_path.stat().st_size

            # Most recent extraction
            cursor = conn.execute("""
                SELECT created_at FROM extraction_results 
                ORDER BY created_at DESC LIMIT 1
            """)
            result = cursor.fetchone()
            stats['last_extraction'] = result[0] if result else None

            return stats


class ProductDatabase:
    """High-level interface for product database operations."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize product database.

        Args:
            db_manager: DatabaseManager instance
        """
        self.db_manager = db_manager or DatabaseManager()
        self.logger = logging.getLogger(__name__)

    def save_product(self, product: Product) -> bool:
        """
        Save a product to the database.

        Args:
            product: Product to save

        Returns:
            True if saved successfully
        """
        try:
            with self.db_manager.get_connection() as conn:
                # Insert or update product
                conn.execute("""
                    INSERT OR REPLACE INTO products (
                        id, product_name, scientific_name, common_names,
                        origin_country, origin_region, origin_location,
                        origin_coordinates, origin_ecosystem, processing_level,
                        additional_info, confidence_score, source_document,
                        extraction_method, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    product.id,
                    product.product_name,
                    product.scientific_name,
                    json.dumps(product.common_names),
                    product.origin.country if product.origin else None,
                    product.origin.region if product.origin else None,
                    product.origin.specific_location if product.origin else None,
                    json.dumps(product.origin.coordinates) if product.origin and product.origin.coordinates else None,
                    product.origin.ecosystem_type if product.origin else None,
                    product.processing_level.value,
                    product.additional_info,
                    product.confidence_score,
                    product.source_document,
                    product.extraction_method,
                    product.created_at.isoformat(),
                    product.updated_at.isoformat()
                ))

                # Delete existing uses and insert new ones
                conn.execute("DELETE FROM product_uses WHERE product_id = ?", (product.id,))

                for use in product.uses:
                    conn.execute("""
                        INSERT INTO product_uses (
                            product_id, category, description, traditional_use,
                            commercial_use, market_value, sustainability_notes
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        product.id,
                        use.category.value,
                        use.description,
                        use.traditional_use,
                        use.commercial_use,
                        use.market_value,
                        use.sustainability_notes
                    ))

                conn.commit()
                self.logger.debug(f"Saved product: {product.product_name}")
                return True

        except Exception as e:
            self.logger.error(f"Failed to save product {product.product_name}: {e}")
            return False

    def get_product(self, product_id: str) -> Optional[Product]:
        """
        Get a product by ID.

        Args:
            product_id: Product ID

        Returns:
            Product instance or None if not found
        """
        try:
            with self.db_manager.get_connection() as conn:
                # Get product data
                cursor = conn.execute("""
                    SELECT * FROM products WHERE id = ?
                """, (product_id,))

                row = cursor.fetchone()
                if not row:
                    return None

                # Get uses
                uses_cursor = conn.execute("""
                    SELECT * FROM product_uses WHERE product_id = ?
                """, (product_id,))

                uses_data = uses_cursor.fetchall()

                # Convert to Product
                return self._row_to_product(row, uses_data)

        except Exception as e:
            self.logger.error(f"Failed to get product {product_id}: {e}")
            return None

    def search_products(
        self,
        name_query: Optional[str] = None,
        country: Optional[str] = None,
        category: Optional[str] = None,
        min_confidence: Optional[float] = None,
        limit: int = 100
    ) -> List[Product]:
        """
        Search products with various filters.

        Args:
            name_query: Product name search query
            country: Country filter
            category: Use category filter
            min_confidence: Minimum confidence score
            limit: Maximum number of results

        Returns:
            List of matching products
        """
        try:
            conditions = []
            params = []

            # Build WHERE clause
            if name_query:
                conditions.append("p.product_name LIKE ?")
                params.append(f"%{name_query}%")

            if country:
                conditions.append("p.origin_country LIKE ?")
                params.append(f"%{country}%")

            if min_confidence is not None:
                conditions.append("p.confidence_score >= ?")
                params.append(min_confidence)

            # Category filter requires join
            query = """
                SELECT DISTINCT p.* FROM products p
            """

            if category:
                query += " JOIN product_uses pu ON p.id = pu.product_id"
                conditions.append("pu.category = ?")
                params.append(category)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY p.confidence_score DESC, p.product_name"
            query += f" LIMIT {limit}"

            with self.db_manager.get_connection() as conn:
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()

                products = []
                for row in rows:
                    # Get uses for each product
                    uses_cursor = conn.execute("""
                        SELECT * FROM product_uses WHERE product_id = ?
                    """, (row['id'],))
                    uses_data = uses_cursor.fetchall()

                    product = self._row_to_product(row, uses_data)
                    if product:
                        products.append(product)

                return products

        except Exception as e:
            self.logger.error(f"Failed to search products: {e}")
            return []

    def save_extraction_result(self, result: ExtractionResult) -> bool:
        """
        Save an extraction result to the database.

        Args:
            result: ExtractionResult to save

        Returns:
            True if saved successfully
        """
        try:
            with self.db_manager.get_connection() as conn:
                # Save extraction result
                metadata = result.metadata
                conn.execute("""
                    INSERT OR REPLACE INTO extraction_results (
                        id, status, overall_confidence, total_products_found,
                        unique_countries, product_categories, extraction_method,
                        model_used, tokens_used, processing_time_seconds,
                        source_file, source_file_size, source_pages, text_length,
                        chunks_processed, total_chunks, errors_encountered,
                        warnings, validation_applied, confidence_threshold,
                        created_at, completed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result.id,
                    result.status.value,
                    result.overall_confidence,
                    result.total_products_found,
                    json.dumps(result.unique_countries),
                    json.dumps(result.product_categories),
                    metadata.extraction_method.value if metadata else None,
                    metadata.model_used if metadata else None,
                    metadata.tokens_used if metadata else None,
                    metadata.processing_time_seconds if metadata else None,
                    metadata.source_file if metadata else None,
                    metadata.source_file_size if metadata else None,
                    metadata.source_pages if metadata else None,
                    metadata.text_length if metadata else None,
                    metadata.chunks_processed if metadata else None,
                    metadata.total_chunks if metadata else None,
                    json.dumps(metadata.errors_encountered) if metadata else None,
                    json.dumps(metadata.warnings) if metadata else None,
                    metadata.validation_applied if metadata else None,
                    metadata.confidence_threshold if metadata else None,
                    result.created_at.isoformat(),
                    result.completed_at.isoformat() if result.completed_at else None
                ))

                # Save products
                for product in result.products:
                    self.save_product(product)

                    # Link product to extraction
                    conn.execute("""
                        INSERT OR REPLACE INTO extraction_products (extraction_id, product_id)
                        VALUES (?, ?)
                    """, (result.id, product.id))

                conn.commit()
                self.logger.info(f"Saved extraction result: {result.id}")
                return True

        except Exception as e:
            self.logger.error(f"Failed to save extraction result {result.id}: {e}")
            return False

    def get_extraction_results(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent extraction results.

        Args:
            limit: Maximum number of results

        Returns:
            List of extraction result summaries
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM extraction_results
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (limit,))

                results = []
                for row in cursor.fetchall():
                    result_dict = dict(row)

                    # Parse JSON fields
                    if result_dict['unique_countries']:
                        result_dict['unique_countries'] = json.loads(result_dict['unique_countries'])
                    if result_dict['product_categories']:
                        result_dict['product_categories'] = json.loads(result_dict['product_categories'])
                    if result_dict['errors_encountered']:
                        result_dict['errors_encountered'] = json.loads(result_dict['errors_encountered'])
                    if result_dict['warnings']:
                        result_dict['warnings'] = json.loads(result_dict['warnings'])

                    results.append(result_dict)

                return results

        except Exception as e:
            self.logger.error(f"Failed to get extraction results: {e}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            with self.db_manager.get_connection() as conn:
                stats = {}

                # Product counts
                cursor = conn.execute("SELECT COUNT(*) FROM products")
                stats['total_products'] = cursor.fetchone()[0]

                # Country distribution
                cursor = conn.execute("""
                    SELECT origin_country, COUNT(*) as count 
                    FROM products 
                    WHERE origin_country IS NOT NULL 
                    GROUP BY origin_country 
                    ORDER BY count DESC
                """)
                stats['countries'] = {row[0]: row[1] for row in cursor.fetchall()}

                # Category distribution
                cursor = conn.execute("""
                    SELECT category, COUNT(*) as count 
                    FROM product_uses 
                    GROUP BY category 
                    ORDER BY count DESC
                """)
                stats['categories'] = {row[0]: row[1] for row in cursor.fetchall()}

                # Confidence distribution
                cursor = conn.execute("""
                    SELECT 
                        COUNT(CASE WHEN confidence_score >= 0.8 THEN 1 END) as high_confidence,
                        COUNT(CASE WHEN confidence_score >= 0.6 AND confidence_score < 0.8 THEN 1 END) as medium_confidence,
                        COUNT(CASE WHEN confidence_score < 0.6 THEN 1 END) as low_confidence
                    FROM products
                """)
                row = cursor.fetchone()
                stats['confidence_distribution'] = {
                    'high': row[0],
                    'medium': row[1], 
                    'low': row[2]
                }

                # Extraction stats
                cursor = conn.execute("SELECT COUNT(*) FROM extraction_results")
                stats['total_extractions'] = cursor.fetchone()[0]

                return stats

        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}")
            return {}

    def _row_to_product(self, row: sqlite3.Row, uses_data: List[sqlite3.Row]) -> Optional[Product]:
        """Convert database row to Product instance."""
        try:
            from src.models import ProductOrigin, ProductUse, ProductCategory, ProcessingLevel

            # Build origin
            origin = None
            if row['origin_country']:
                coordinates = None
                if row['origin_coordinates']:
                    coordinates = json.loads(row['origin_coordinates'])

                origin = ProductOrigin(
                    country=row['origin_country'],
                    region=row['origin_region'],
                    specific_location=row['origin_location'],
                    coordinates=coordinates,
                    ecosystem_type=row['origin_ecosystem']
                )

            # Build uses
            uses = []
            for use_row in uses_data:
                use = ProductUse(
                    category=ProductCategory(use_row['category']),
                    description=use_row['description'],
                    traditional_use=bool(use_row['traditional_use']),
                    commercial_use=bool(use_row['commercial_use']),
                    market_value=use_row['market_value'],
                    sustainability_notes=use_row['sustainability_notes']
                )
                uses.append(use)

            # Parse common names
            common_names = []
            if row['common_names']:
                common_names = json.loads(row['common_names'])

            # Create product
            product = Product(
                id=row['id'],
                product_name=row['product_name'],
                scientific_name=row['scientific_name'],
                common_names=common_names,
                origin=origin,
                uses=uses,
                processing_level=ProcessingLevel(row['processing_level']),
                additional_info=row['additional_info'] or "",
                confidence_score=row['confidence_score'],
                source_document=row['source_document'],
                extraction_method=row['extraction_method'],
                created_at=datetime.fromisoformat(row['created_at']),
                updated_at=datetime.fromisoformat(row['updated_at'])
            )

            return product

        except Exception as e:
            self.logger.error(f"Failed to convert row to product: {e}")
            return None
