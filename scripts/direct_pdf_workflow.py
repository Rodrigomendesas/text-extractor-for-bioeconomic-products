"""
Direct PDF workflow that sends PDF files directly to OpenAI API without text extraction.
This simplifies the workflow by eliminating the text extraction step and letting the API handle it.
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
    """Validate all prerequisites before running the workflow."""
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

    # Check if PDF files exist
    pdf_dir = Path("data/input/pdfs")
    if not pdf_dir.exists() or not list(pdf_dir.glob("*.pdf")):
        print("❌ Nenhum arquivo PDF encontrado em data/input/pdfs!")
        return False

    return True


def direct_pdf_workflow(pdf_path: Path) -> Optional[Dict[str, Any]]:
    """Process a PDF file by sending it directly to OpenAI API."""
    print(f"🌳 Fluxo Direto Pan-Amazônico com PDF: {pdf_path.name}")
    print("=" * 70)

    try:
        # Initialize components
        print("🔧 Inicializando componentes...")
        client = OpenAIClient()
        pm = PanAmazonPromptManager()
        parser = PanAmazonResponseParser()

        # Get the extraction prompt
        from src.llm.prompt_manager import PromptType
        extraction_prompt = pm.get_prompt(PromptType.SOCIOBIODIVERSITY_EXTRACTION)

        # Format the prompts
        source_name = pdf_path.stem
        system_prompt = extraction_prompt.system_prompt
        user_prompt = extraction_prompt.user_template.format(
            text="[O conteúdo será extraído diretamente do arquivo PDF]",
            source_name=source_name
        )

        print(f"📄 Enviando arquivo PDF diretamente para a API: {pdf_path.name}...")

        # Send the file directly to OpenAI API
        response = client.file_completion(
            file_path=str(pdf_path),
            system_prompt=system_prompt,
            user_prompt=user_prompt
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
            ""  # No original text since we're sending the file directly
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
            "metadados_pdf": {
                "nome_arquivo": pdf_path.name,
                "metodo_extracao": "direct_api_upload"
            },
            "custos": {
                "tokens_entrada": response.usage_details["prompt_tokens"],
                "tokens_saida": response.usage_details["completion_tokens"],
                "tokens_total": response.usage_details["total_tokens"],
                "custo_usd": cost,
                "modelo": response.model
            }
        }

        # Save to file with error handling
        output_file = Path(f"data/output/direct_pdf_{pdf_path.stem}.json")
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"✅ Resultado salvo em: {output_file}")
        except (OSError, IOError) as e:
            print(f"⚠️ Erro ao salvar arquivo: {e}")
            # Continue execution even if file save fails

        print(f"\n🎉 TESTE CONCLUÍDO COM SUCESSO!")
        print(f"Sistema validado com envio direto de PDF!")

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
    """Run direct PDF workflow test."""
    print("Validando configuração...")

    if not validate_prerequisites():
        return False

    # Get a sample PDF file
    pdf_dir = Path("data/input/pdfs")
    pdf_files = list(pdf_dir.glob("*.pdf"))

    # Choose a smaller PDF file for testing
    pdf_files.sort(key=lambda p: p.stat().st_size)
    sample_pdf = pdf_files[0]  # Use the smallest PDF file

    print(f"Usando arquivo de teste: {sample_pdf.name} ({sample_pdf.stat().st_size / 1024:.1f} KB)")

    # Run test
    result = direct_pdf_workflow(sample_pdf)

    if result is not None:
        print("\n✅ Sistema pronto para processar documentos PDF da Pan-Amazônia diretamente!")
        return True
    else:
        print("\n❌ Teste falhou. Verifique a configuração.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
