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

        system_prompt = """Você é um analista especializado em bioeconomia amazônica. Seu papel é extrair informações de textos técnicos e relatórios para mapear produtos da bioeconomia da sociobiodiversidade com uso produtivo em diferentes países. Voce deve realizar uma verificação dupla linha por linha do texto.

Com base no texto, identifique os produtos da bioeconomia da sociobiodiversidade que estejam presentes no texto. A análise deve priorizar produtos que possuam valor econômico, cultural ou ambiental, mesmo que de forma incipiente, em nível local ou regional. Extraia e liste os produtos que estejam literalmente mencionados ou claramente inferidos no texto fornecido. Não invente, preencha apenas com base no conteúdo visível. Se a informação não estiver presente, escreva "não especificado".

Considere como produtos da sociobiodiversidade:
1. Plantas e seus derivados (frutas, sementes, óleos, resinas, fibras, madeiras, etc.)
2. Animais e seus derivados (mel, carne, couro, etc.)
3. Fungos e microorganismos com uso produtivo
4. Produtos processados derivados de recursos naturais da região

Caso o nome do produto não esteja literal, mas o contexto permita inferir com segurança (ex: descrição de uso tradicional, habitat, parte da planta, aplicação típica), inclua na tabela e explique no campo 'Trecho justificativo' o motivo da inferência.

Considere variações linguísticas e nomes locais de produtos da sociobiodiversidade, mesmo que não correspondam ao nome científico completo.

Critérios de inclusão:

Inclua produtos com nome popular ou científico mencionados diretamente no texto, OU produtos claramente inferidos com base em descrições específicas, desde que: A descrição permita reconhecer o produto com base no conteúdo do texto. Você inclua obrigatoriamente um trecho do texto que comprove a menção do produto ou justifique a inferência.

Em caso de nomes ambíguos, com possíveis sinonímias regionais (ex: 'ungurahui', 'patauá'), inclua somente se houver indicação no texto que permita identificar a espécie ou produto. Se não for possível, mas o produto for relevante, inclua com uma observação sobre a ambiguidade.

É importante que cada produto listado tenha um trecho do texto que o mencione ou justifique a inferência. Se o produto for inferido, o trecho deve conter elementos descritivos suficientes para permitir identificação. Em caso de dúvida, mas havendo indícios suficientes, inclua o produto com uma confiança menor.

Quando o texto mencionar categorias de produtos (ex: "óleos da floresta", "resinas", "fibras vegetais"), inclua-os como produtos específicos se o contexto permitir identificar seu uso produtivo, mesmo que não mencione espécies específicas.

BUSQUE DIVERSIDADE de produtos. Não se limite apenas aos mais óbvios ou mais frequentes. Procure identificar diferentes tipos de produtos mencionados no texto, mesmo que apareçam apenas uma vez.

EXCLUSÕES:

Produtos extremamente genéricos sem qualquer especificação ou contexto (ex: "recursos naturais" sem nenhum contexto adicional);

Atividades sem vínculo com produtos específicos (ex: turismo, quando não menciona produtos);"""

        user_template = """Com base no texto, identifique os produtos da bioeconomia da sociobiodiversidade presentes no texto. A análise deve priorizar produtos que possuam valor econômico, cultural ou ambiental, em nível local ou regional. Extraia e liste os produtos que estejam mencionados ou claramente inferidos no texto fornecido. Se a informação não estiver presente, escreva "não especificado".

Considere como produtos da sociobiodiversidade:
1. Plantas e seus derivados (frutas, sementes, óleos, resinas, fibras, madeiras, etc.)
2. Animais e seus derivados (mel, carne, couro, etc.)
3. Fungos e microorganismos com uso produtivo
4. Produtos processados derivados de recursos naturais da região

Caso o nome do produto não esteja literal, mas o contexto permita inferir (ex: descrição de uso tradicional, habitat, parte da planta, aplicação típica), inclua na tabela e explique no campo 'Trecho justificativo' o motivo da inferência.

BUSQUE DIVERSIDADE de produtos. Não se limite apenas aos mais óbvios ou mais frequentes. Procure identificar diferentes tipos de produtos mencionados no texto, mesmo que apareçam apenas uma vez.

Quando o texto mencionar categorias de produtos (ex: "óleos da floresta", "resinas", "fibras vegetais"), inclua-os como produtos específicos se o contexto permitir identificar seu uso produtivo.

Para cada produto encontrado, preencha os seguintes campos:

- Nome popular (como consta no texto, manter o idioma original)
- Nome científico (quando mencionado no texto)
- País(es) onde o produto é produzido: Indique se houver menção direta ou se puder ser inferido do contexto
- Região onde o produto é produzido: Indique se houver menção direta ou se puder ser inferido do contexto
- Comunidade ou povo associado (se houver menção)
- Tipos de uso (alimentar, medicinal, cosmético, artesanal, construção, têxtil, tintorial, ritual, outro)
- Categoria do produto (fruta, semente, óleo, madeira, fibra, etc.)
- Informações adicionais (qualquer outra informação relevante sobre o produto)
- Trecho justificativo: um trecho do texto que mencione ou sugira o produto
- Fonte: "{source_name}"

Etapa extra obrigatória:

Antes de apresentar a tabela final, verifique se você incluiu uma diversidade de produtos. Se você perceber que está listando apenas um ou dois tipos de produtos repetidamente, revise o texto para identificar outros produtos mencionados.

Responda APENAS com JSON no seguinte formato:
{{
  "produtos": [
    {{
      "nome_popular": "Nome do produto",
      "nome_cientifico": "Nome científico se disponível",
      "paises": ["Brasil", "Peru"],
      "regiao": ["Amazônia", "Norte do Peru"],
      "comunidade": "Nome da comunidade ou povo",
      "tipos_uso": ["alimentar", "medicinal"],
      "categoria": "Tipo de produto",
      "observacoes": "Outras informações adicionais",
      "trecho_justificativo": "Trecho do texto que menciona ou sugere o produto",
      "fonte": "{source_name}",
      "confianca": 0.9
    }}
  ],
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
