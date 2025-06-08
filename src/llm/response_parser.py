"""
Parse and validate LLM responses for Pan-Amazon sociobiodiversity extraction.
Optimized for the client's proven data structure.
"""

import json
import re
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ParseError(Exception):
    """Custom exception for parsing errors."""
    pass


@dataclass
class SociobiodiversityProduct:
    """Pan-Amazon sociobiodiversity product data structure."""
    nome_popular: str
    nome_cientifico: Optional[str] = None
    paises: List[str] = None
    tipos_uso: List[str] = None
    fonte: str = ""
    confianca: float = 0.0

    def __post_init__(self):
        if self.paises is None:
            self.paises = []
        if self.tipos_uso is None:
            self.tipos_uso = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "nome_popular": self.nome_popular,
            "nome_cientifico": self.nome_cientifico,
            "paises": self.paises,
            "tipos_uso": self.tipos_uso,
            "fonte": self.fonte,
            "confianca": self.confianca
        }


@dataclass
class ExtractionResult:
    """Complete extraction result for Pan-Amazon analysis."""
    produtos: List[SociobiodiversityProduct]
    resumo: str
    idioma_detectado: str
    total_produtos: int
    fonte_documento: str
    metadados_processamento: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "produtos": [p.to_dict() for p in self.produtos],
            "resumo": self.resumo,
            "idioma_detectado": self.idioma_detectado,
            "total_produtos": self.total_produtos,
            "fonte_documento": self.fonte_documento,
            "metadados_processamento": self.metadados_processamento
        }


class PanAmazonResponseParser:
    """Parse and validate responses for Pan-Amazon sociobiodiversity extraction."""

    def __init__(self):
        self.valid_countries = [
            "brasil", "bolívia", "colômbia", "equador", "peru",
            "brazil", "bolivia", "colombia", "ecuador"
        ]
        self.valid_use_types = [
            "alimentar", "medicinal", "cosmético", "artesanal", 
            "construção", "têxtil", "tintorial", "ritual", "outro"
        ]
        self.valid_languages = {"pt", "es", "en", "outro"}

    def parse_extraction_response(
        self, 
        response_content: str, 
        source_name: str = "documento",
        original_text: str = ""
    ) -> ExtractionResult:
        """
        Parse extraction response into structured data.

        Args:
            response_content: Raw LLM response
            source_name: Name of the source document
            original_text: Original text that was processed

        Returns:
            ExtractionResult with parsed data
        """
        try:
            # Clean and extract JSON
            json_content = self._extract_json(response_content)

            # Parse JSON
            data = json.loads(json_content)

            # Validate structure
            self._validate_response_structure(data)

            # Parse products
            produtos = self._parse_produtos(data.get("produtos", []))

            # Extract metadata
            resumo = data.get("resumo", "")
            idioma_detectado = data.get("idioma_detectado", "pt")
            total_produtos = data.get("total_produtos", len(produtos))

            # Validate language
            if idioma_detectado not in self.valid_languages:
                idioma_detectado = "outro"

            # Update total if not matching
            if total_produtos != len(produtos):
                total_produtos = len(produtos)

            # Create processing metadata
            metadados_processamento = {
                "total_produtos_extraidos": len(produtos),
                "produtos_alta_confianca": len([p for p in produtos if p.confianca >= 0.7]),
                "paises_encontrados": list(set(
                    pais.lower() for produto in produtos 
                    for pais in produto.paises
                )),
                "tipos_uso_encontrados": list(set(
                    uso for produto in produtos 
                    for uso in produto.tipos_uso
                )),
                "produtos_com_nome_cientifico": sum(1 for p in produtos if p.nome_cientifico),
                "tamanho_texto_original": len(original_text),
                "tamanho_resposta": len(response_content),
                "sucesso_parsing": True
            }

            return ExtractionResult(
                produtos=produtos,
                resumo=resumo,
                idioma_detectado=idioma_detectado,
                total_produtos=total_produtos,
                fonte_documento=source_name,
                metadados_processamento=metadados_processamento
            )

        except Exception as e:
            logger.error(f"Failed to parse extraction response: {e}")
            logger.debug(f"Response content: {response_content[:500]}...")

            # Return empty result with error metadata
            return ExtractionResult(
                produtos=[],
                resumo="Erro no processamento",
                idioma_detectado="desconhecido",
                total_produtos=0,
                fonte_documento=source_name,
                metadados_processamento={
                    "erro": str(e),
                    "sucesso_parsing": False,
                    "tamanho_texto_original": len(original_text),
                    "tamanho_resposta": len(response_content)
                }
            )

    def _extract_json(self, content: str) -> str:
        """Extract JSON from response content."""
        # Remove markdown code blocks
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```\s*$', '', content)

        # Try to find JSON object
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json_match.group(0)

        # If no JSON found, try to clean the content
        content = content.strip()
        if content.startswith('{') and content.endswith('}'):
            return content

        raise ParseError(f"No valid JSON found in response")

    def _validate_response_structure(self, data: Dict[str, Any]) -> None:
        """Validate the basic structure of the response."""
        required_fields = ["produtos"]

        for field in required_fields:
            if field not in data:
                raise ParseError(f"Missing required field: {field}")

        if not isinstance(data["produtos"], list):
            raise ParseError("Produtos field must be a list")

    def _parse_produtos(self, produtos_data: List[Dict[str, Any]]) -> List[SociobiodiversityProduct]:
        """Parse products from JSON data."""
        produtos = []

        for i, produto_data in enumerate(produtos_data):
            try:
                produto = self._parse_single_produto(produto_data)
                if produto:
                    produtos.append(produto)
            except Exception as e:
                logger.warning(f"Failed to parse product {i}: {e}")
                continue

        return produtos

    def _parse_single_produto(self, data: Dict[str, Any]) -> Optional[SociobiodiversityProduct]:
        """Parse a single product from JSON data."""
        # Validate required fields
        if "nome_popular" not in data or not data["nome_popular"]:
            logger.warning("Product missing nome_popular field")
            return None

        # Parse and validate countries
        paises = data.get("paises", [])
        if isinstance(paises, str):
            paises = [paises]
        elif not isinstance(paises, list):
            paises = []

        # Clean and validate countries
        paises_validos = []
        for pais in paises:
            pais_clean = str(pais).strip()
            if pais_clean and self._validate_country(pais_clean):
                paises_validos.append(pais_clean)

        # Parse and validate use types
        tipos_uso = data.get("tipos_uso", [])
        if isinstance(tipos_uso, str):
            tipos_uso = [tipos_uso]
        elif not isinstance(tipos_uso, list):
            tipos_uso = []

        # Clean and validate use types
        tipos_uso_validos = []
        for tipo in tipos_uso:
            tipo_clean = str(tipo).strip().lower()
            if tipo_clean:
                tipo_normalizado = self._normalize_use_type(tipo_clean)
                if tipo_normalizado:
                    tipos_uso_validos.append(tipo_normalizado)

        # Validate confidence
        confianca = data.get("confianca", 0.0)
        try:
            confianca = float(confianca)
            confianca = max(0.0, min(1.0, confianca))  # Clamp between 0 and 1
        except (ValueError, TypeError):
            confianca = 0.0

        return SociobiodiversityProduct(
            nome_popular=str(data["nome_popular"]).strip(),
            nome_cientifico=data.get("nome_cientifico", "").strip() or None,
            paises=paises_validos,
            tipos_uso=tipos_uso_validos,
            fonte=data.get("fonte", "").strip(),
            confianca=confianca
        )

    def _validate_country(self, country: str) -> bool:
        """Check if country is valid for Pan-Amazon region."""
        country_lower = country.lower().strip()

        # Direct matches
        if country_lower in self.valid_countries:
            return True

        # Partial matches for common variations
        country_variations = {
            "brazil": True,
            "brasil": True,
            "bolivia": True,
            "bolívia": True,
            "colombia": True,
            "colômbia": True,
            "ecuador": True,
            "equador": True,
            "peru": True,
            "perú": True
        }

        return country_variations.get(country_lower, False)

    def _normalize_use_type(self, use_type: str) -> Optional[str]:
        """Normalize use type to standard categories."""
        use_lower = use_type.lower().strip()

        # Map common variations to standard types
        use_mapping = {
            "alimentício": "alimentar",
            "alimenticia": "alimentar",
            "comida": "alimentar",
            "alimento": "alimentar",
            "remédio": "medicinal",
            "medicina": "medicinal",
            "farmacêutico": "medicinal",
            "medicamento": "medicinal",
            "beleza": "cosmético",
            "cosmetico": "cosmético",
            "artesanato": "artesanal",
            "artesã": "artesanal",
            "construir": "construção",
            "construcao": "construção",
            "madeira": "construção",
            "tecido": "têxtil",
            "textil": "têxtil",
            "fibra": "têxtil",
            "cor": "tintorial",
            "tinta": "tintorial",
            "corante": "tintorial",
            "cerimônia": "ritual",
            "cerimonia": "ritual",
            "religioso": "ritual",
            "sagrado": "ritual",
            "espiritual": "ritual"
        }

        # Try direct mapping first
        normalized = use_mapping.get(use_lower, use_lower)

        # Check if it's in valid types
        if normalized in self.valid_use_types:
            return normalized

        # If not found, return "outro"
        return "outro"

    def parse_classification_response(self, response_content: str) -> str:
        """Parse document classification response."""
        content = response_content.strip().lower()

        if "sim" in content:
            return "sim"
        elif "não" in content or "nao" in content:
            return "não"
        elif "talvez" in content:
            return "talvez"
        else:
            return "talvez"  # Default to maybe if unclear

    def parse_language_detection(self, response_content: str) -> str:
        """Parse language detection response."""
        content = response_content.strip().lower()

        for lang in self.valid_languages:
            if lang in content:
                return lang

        return "outro"

    def validate_extraction_quality(self, result: ExtractionResult) -> Dict[str, Any]:
        """Assess the quality of extraction results."""
        produtos = result.produtos

        if not produtos:
            return {
                "qualidade_geral": 0.0,
                "total_produtos": 0,
                "confianca_media": 0.0,
                "produtos_com_nome_cientifico": 0,
                "produtos_com_paises": 0,
                "produtos_com_usos": 0,
                "produtos_alta_confianca": 0,
                "produtos_baixa_confianca": 0,
                "observacoes": ["Nenhum produto extraído"]
            }

        quality_metrics = {
            "total_produtos": len(produtos),
            "confianca_media": sum(p.confianca for p in produtos) / len(produtos),
            "produtos_com_nome_cientifico": sum(1 for p in produtos if p.nome_cientifico),
            "produtos_com_paises": sum(1 for p in produtos if p.paises),
            "produtos_com_usos": sum(1 for p in produtos if p.tipos_uso),
            "produtos_alta_confianca": sum(1 for p in produtos if p.confianca >= 0.7),
            "produtos_baixa_confianca": sum(1 for p in produtos if p.confianca < 0.5),
        }

        # Calculate quality score
        completeness_score = (
            quality_metrics["produtos_com_nome_cientifico"] / len(produtos) * 0.3 +
            quality_metrics["produtos_com_paises"] / len(produtos) * 0.3 +
            quality_metrics["produtos_com_usos"] / len(produtos) * 0.2 +
            quality_metrics["confianca_media"] * 0.2
        )

        quality_metrics["qualidade_geral"] = round(completeness_score, 2)

        # Add observations
        observacoes = []
        if quality_metrics["produtos_baixa_confianca"] > len(produtos) * 0.3:
            observacoes.append("Muitos produtos com baixa confiança")
        if quality_metrics["produtos_com_nome_cientifico"] < len(produtos) * 0.5:
            observacoes.append("Poucos nomes científicos identificados")
        if quality_metrics["confianca_media"] >= 0.8:
            observacoes.append("Boa qualidade geral dos dados")

        quality_metrics["observacoes"] = observacoes

        return quality_metrics
