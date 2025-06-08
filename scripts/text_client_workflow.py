"""
Test script that replicates the client's proven workflow.
"""
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any

# Add repository root and src to path
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))
sys.path.append(str(repo_root / "src"))

from config.settings import settings, config_manager
from src.llm.openai_client import OpenAIClient
from src.llm.prompt_manager import PanAmazonPromptManager
from src.llm.response_parser import PanAmazonResponseParser


def validate_prerequisites() -> bool:
    """Validate all prerequisites before running the test."""
    # Check API key
    validation = config_manager.validate_api_keys()
    if not validation["openai"]:
        print("❌ Chave da API OpenAI não configurada!")
        return False

    # Ensure output directory exists
    output_dir = Path("data/output")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"❌ Erro ao criar diretório de saída: {e}")
        return False

    return True


def test_client_workflow() -> Optional[Dict[str, Any]]:
    """Test the exact workflow the client used successfully."""
    print("🌳 Teste do Fluxo Pan-Amazônico - Produtos da Sociobiodiversidade")
    print("=" * 70)

    # Sample text similar to what the client would process
    sample_text = """
    A região amazônica do Brasil é rica em produtos da sociobiodiversidade. 
    O açaí (Euterpe oleracea) é amplamente consumido como alimento no Pará e Amapá, 
    sendo também utilizado na fabricação de cosméticos. 
    A castanha-do-pará (Bertholletia excelsa) é coletada de forma sustentável 
    por comunidades tradicionais no Acre e Rondônia, sendo exportada como alimento nutritivo.
    Na Colômbia, a copaíba (Copaifera officinalis) é utilizada na medicina tradicional 
    para tratamento de feridas e inflamações. No Peru, a maca (Lepidium meyenii) 
    é consumida como alimento funcional e suplemento nutricional.
    O buriti (Mauritia flexuosa) é utilizado no Equador para artesanato, 
    especialmente na confecção de cestas e chapéus tradicionais.
    """
    source_name = "Relatório Sociobiodiversidade Pan-Amazônica 2024"

    try:
        # Initialize components
        print("🔧 Inicializando componentes...")
        client = OpenAIClient()
        pm = PanAmazonPromptManager()
        parser = PanAmazonResponseParser()

        # Format prompt using client's approach
        print("📝 Formatando prompt...")
        formatted_prompt = pm.format_extraction_prompt(sample_text, source_name)
        print(f"📊 Prompt System: {len(formatted_prompt['system'])} caracteres")
        print(f"📊 Prompt User: {len(formatted_prompt['user'])} caracteres")

        # Make request with error handling
        print("🤖 Fazendo requisição para OpenAI...")
        response = client.system_user_completion(
            formatted_prompt["system"],
            formatted_prompt["user"]
        )

        # Validate response structure
        if not hasattr(response, 'usage_details') or not response.usage_details:
            raise ValueError("Resposta da API incompleta - faltam detalhes de uso")

        required_keys = ["prompt_tokens", "completion_tokens", "total_tokens"]
        missing_keys = [key for key in required_keys if key not in response.usage_details]
        if missing_keys:
            raise ValueError(f"Resposta da API incompleta - faltam chaves: {missing_keys}")

        print(f"✅ Resposta recebida em {response.processing_time:.2f}s")
        print(f"📊 Tokens utilizados: {response.tokens_used}")
        print(f"📊 Modelo: {response.model}")

        # Calculate cost with safety check
        try:
            cost = settings.estimate_cost(
                response.usage_details["prompt_tokens"],
                response.usage_details["completion_tokens"],
                response.model
            )
            print(f"💰 Custo estimado: ${cost:.6f}")

            # Cost safety check
            if cost > settings.cost_alert_threshold_usd:
                print(f"⚠️ Aviso: Custo acima do limite configurado (${settings.cost_alert_threshold_usd})")

        except Exception as e:
            print(f"⚠️ Erro ao calcular custo: {e}")
            cost = 0.0

        # Parse response
        print("🔍 Analisando resposta...")
        result = parser.parse_extraction_response(
            response.content, 
            source_name, 
            sample_text
        )

        # Display results
        print(f"\n📊 RESULTADOS DA EXTRAÇÃO")
        print(f"Total de produtos encontrados: {result.total_produtos}")
        print(f"Idioma detectado: {result.idioma_detectado}")
        print(f"Resumo: {result.resumo}")

        print(f"\n🌿 PRODUTOS EXTRAÍDOS:")
        for i, produto in enumerate(result.produtos, 1):
            print(f"\n{i}. {produto.nome_popular}")
            if produto.nome_cientifico:
                print(f"   Científico: {produto.nome_cientifico}")
            if produto.paises:
                print(f"   Países: {', '.join(produto.paises)}")
            if produto.tipos_uso:
                print(f"   Usos: {', '.join(produto.tipos_uso)}")
            print(f"   Confiança: {produto.confianca}")
            print(f"   Fonte: {produto.fonte}")

        # Quality assessment
        quality = parser.validate_extraction_quality(result)
        print(f"\n📈 AVALIAÇÃO DE QUALIDADE:")
        print(f"Qualidade geral: {quality['qualidade_geral']}")
        print(f"Confiança média: {quality['confianca_media']:.2f}")
        print(f"Produtos com nome científico: {quality['produtos_com_nome_cientifico']}")
        print(f"Produtos com países: {quality['produtos_com_paises']}")
        if quality['observacoes']:
            print(f"Observações: {', '.join(quality['observacoes'])}")

        # Export sample result
        print(f"\n💾 Exportando resultado...")
        output_data = {
            "resultado_extracao": result.to_dict(),
            "avaliacao_qualidade": quality,
            "custos": {
                "tokens_entrada": response.usage_details["prompt_tokens"],
                "tokens_saida": response.usage_details["completion_tokens"],
                "tokens_total": response.usage_details["total_tokens"],
                "custo_usd": cost,
                "modelo": response.model
            }
        }

        # Save to file with error handling
        output_file = Path("data/output/teste_client_workflow.json")
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"✅ Resultado salvo em: {output_file}")
        except (OSError, IOError) as e:
            print(f"⚠️ Erro ao salvar arquivo: {e}")
            # Continue execution even if file save fails

        print(f"\n🎉 TESTE CONCLUÍDO COM SUCESSO!")
        print(f"Sistema validado com o fluxo exato da cliente!")

        return output_data

    except KeyError as e:
        print(f"❌ Erro de configuração: campo ausente {e}")
        return None
    except ValueError as e:
        print(f"❌ Erro de validação: {e}")
        return None
    except ConnectionError as e:
        print(f"❌ Erro de conexão com OpenAI: {e}")
        return None
    except Exception as e:
        print(f"❌ Erro inesperado no teste: {e}")
        import traceback
        if settings.debug_mode:
            traceback.print_exc()
        return None


def main() -> bool:
    """Run client workflow test."""
    print("Validando configuração...")

    if not validate_prerequisites():
        return False

    # Run test
    result = test_client_workflow()

    if result is not None:
        print("\n✅ Sistema pronto para processar documentos da Pan-Amazônia!")
        return True
    else:
        print("\n❌ Teste falhou. Verifique a configuração.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
