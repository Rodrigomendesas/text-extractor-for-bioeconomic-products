"""
Prompt management optimized for Pan-Amazon sociobiodiversity product extraction.
Based on proven client prompt for Brazil, Bolivia, Colombia, Ecuador, and Peru.
"""

import json
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass

from config.settings import settings


class PromptType(str, Enum):
    """Types of prompts for different extraction tasks."""
    SOCIOBIODIVERSITY_EXTRACTION = "sociobiodiversity_extraction"
    LANGUAGE_DETECTION = "language_detection"
    DOCUMENT_CLASSIFICATION = "document_classification"


@dataclass
class ExtractionPrompt:
    """Container for extraction prompt configuration."""
    system_prompt: str
    user_template: str
    language: str
    max_tokens: int
    temperature: float


class PanAmazonPromptManager:
    """
    Specialized prompt manager for Pan-Amazon sociobiodiversity products.
    Focused on Brazil, Bolivia, Colombia, Ecuador, and Peru.
    """
    
    def __init__(self):
        self.target_countries = ["Brasil", "Bolívia", "Colômbia", "Equador", "Peru"]
        self.use_types = [
            "alimentar", "medicinal", "cosmético", "artesanal", 
            "construção", "têxtil", "tintorial", "ritual", "outro"
        ]
        self.prompts = self._initialize_prompts()
    
    def _initialize_prompts(self) -> Dict[str, ExtractionPrompt]:
        """Initialize prompt templates based on client requirements."""
        return {
            PromptType.SOCIOBIODIVERSITY_EXTRACTION: self._get_sociobiodiversity_prompt(),
            PromptType.LANGUAGE_DETECTION: self._get_language_detection_prompt(),
            PromptType.DOCUMENT_CLASSIFICATION: self._get_classification_prompt()
        }
    
    def _get_sociobiodiversity_prompt(self) -> ExtractionPrompt:
        """Get the main sociobiodiversity extraction prompt (client's proven prompt)."""
        
        system_prompt = """Você é um analista especializado em bioeconomia da Pan-Amazônia. 

Seu objetivo é extrair dados estruturados de textos técnicos, com foco em produtos da sociobiodiversidade utilizados nos países: Brasil, Bolívia, Colômbia, Equador e Peru.

INSTRUÇÕES IMPORTANTES:
- Extraia APENAS produtos explicitamente mencionados no texto
- Foque em produtos da sociobiodiversidade (plantas, animais, fungos, minerais)
- Inclua nomes científicos quando disponíveis
- Identifique países da Pan-Amazônia mencionados
- Classifique os tipos de uso corretamente
- Seja preciso e conciso
- Responda APENAS com JSON válido"""

        user_template = """A partir do texto abaixo, identifique e liste os produtos citados, preenchendo os seguintes campos para cada item encontrado:

- Nome popular do produto
- Nome científico (se disponível)  
- País(es) onde é mencionado ou utilizado
- Tipo de uso (alimentar, medicinal, cosmético, artesanal, construção, têxtil, tintorial, ritual, outro)
- Fonte (nome do texto)

TEXTO:
{text}

FONTE DO TEXTO:
{source_name}

Responda APENAS com JSON no seguinte formato:
{{
  "produtos": [
    {{
      "nome_popular": "Nome do produto",
      "nome_cientifico": "Nome científico se disponível",
      "paises": ["Brasil", "Peru"],
      "tipos_uso": ["alimentar", "medicinal"],
      "fonte": "{source_name}",
      "confianca": 0.9
    }}
  ],
  "resumo": "Breve resumo do conteúdo analisado",
  "idioma_detectado": "pt|es|en",
  "total_produtos": 0
}}

JSON:"""

        return ExtractionPrompt(
            system_prompt=system_prompt,
            user_template=user_template,
            language="pt",
            max_tokens=settings.openai_max_tokens,
            temperature=settings.openai_temperature
        )
    
    def _get_language_detection_prompt(self) -> ExtractionPrompt:
        """Simple language detection for Pan-Amazon region."""
        system_prompt = "Detecte o idioma principal do texto. Responda apenas com o código do idioma."

        user_template = """Detecte o idioma. Responda APENAS com: pt, es, en, ou outro

TEXTO: {text}

Idioma:"""

        return ExtractionPrompt(
            system_prompt=system_prompt,
            user_template=user_template,
            language="multilingual",
            max_tokens=5,
            temperature=0.0
        )
    
    def _get_classification_prompt(self) -> ExtractionPrompt:
        """Classify if document contains sociobiodiversity information."""
        system_prompt = "Classifique se o texto contém informações sobre produtos da sociobiodiversidade da Pan-Amazônia."

        user_template = """Este texto contém informações sobre produtos da sociobiodiversidade (plantas, animais, recursos naturais) da Pan-Amazônia?

Responda APENAS: sim, não, ou talvez

TEXTO: {text}

Contém produtos da sociobiodiversidade:"""

        return ExtractionPrompt(
            system_prompt=system_prompt,
            user_template=user_template,
            language="pt",
            max_tokens=5,
            temperature=0.0
        )
    
    def get_prompt(self, prompt_type: PromptType) -> ExtractionPrompt:
        """Get a specific prompt by type."""
        return self.prompts[prompt_type]
    
    def format_extraction_prompt(self, text: str, source_name: str = "documento") -> Dict[str, str]:
        """Format the main extraction prompt with text and source."""
        prompt = self.get_prompt(PromptType.SOCIOBIODIVERSITY_EXTRACTION)
        
        user_content = prompt.user_template.format(
            text=text,
            source_name=source_name
        )
        
        return {
            "system": prompt.system_prompt,
            "user": user_content
        }
    
    def format_classification_prompt(self, text: str) -> Dict[str, str]:
        """Format classification prompt."""
        prompt = self.get_prompt(PromptType.DOCUMENT_CLASSIFICATION)
        
        return {
            "system": prompt.system_prompt,
            "user": prompt.user_template.format(text=text)
        }
    
    def format_language_detection_prompt(self, text: str) -> Dict[str, str]:
        """Format language detection prompt."""
        prompt = self.get_prompt(PromptType.LANGUAGE_DETECTION)
        
        return {
            "system": prompt.system_prompt,
            "user": prompt.user_template.format(text=text)
        }
    
    def get_token_estimate(self, prompt_type: PromptType, text_length: int) -> int:
        """Estimate token usage for a prompt type."""
        prompt = self.get_prompt(prompt_type)
        
        # Rough estimation: system + template + text
        base_tokens = len(prompt.system_prompt + prompt.user_template) // 4
        text_tokens = text_length // 4
        
        return base_tokens + text_tokens
    
    def validate_country(self, country: str) -> bool:
        """Check if country is in Pan-Amazon target list."""
        country_lower = country.lower()
        target_countries_lower = [c.lower() for c in self.target_countries]
        
        # Check exact matches and common variations
        country_variations = {
            "brazil": "brasil",
            "bolivia": "bolívia", 
            "colombia": "colômbia",
            "ecuador": "equador",
            "peru": "peru"
        }
        
        normalized = country_variations.get(country_lower, country_lower)
        return normalized in target_countries_lower
    
    def validate_use_type(self, use_type: str) -> str:
        """Validate and normalize use type."""
        use_lower = use_type.lower().strip()
        
        # Map common variations to standard types
        use_mapping = {
            "alimentício": "alimentar",
            "comida": "alimentar",
            "remédio": "medicinal",
            "medicina": "medicinal",
            "beleza": "cosmético",
            "artesanato": "artesanal",
            "construir": "construção",
            "tecido": "têxtil",
            "cor": "tintorial",
            "tinta": "tintorial",
            "cerimônia": "ritual",
            "religioso": "ritual"
        }
        
        normalized = use_mapping.get(use_lower, use_lower)
        
        if normalized in self.use_types:
            return normalized
        else:
            return "outro"
    
    def get_extraction_guidelines(self) -> Dict[str, Any]:
        """Get extraction guidelines for documentation."""
        return {
            "target_countries": self.target_countries,
            "valid_use_types": self.use_types,
            "focus": "Produtos da sociobiodiversidade da Pan-Amazônia",
            "required_fields": [
                "nome_popular",
                "nome_cientifico", 
                "paises",
                "tipos_uso",
                "fonte"
            ],
            "optional_fields": [
                "confianca",
                "observacoes"
            ]
        }