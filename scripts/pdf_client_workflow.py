"""
Test script that processes a PDF file and extracts bioeconomic products.
This extends the client workflow to work with actual PDF files.
It now also extracts and processes visual elements (images, charts, tables) from PDFs.
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
from src.core.pdf_processor import PDFProcessor


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

    # Check if PDF files exist
    pdf_dir = Path("data/input/pdfs")
    if not pdf_dir.exists() or not list(pdf_dir.glob("*.pdf")):
        print("❌ Nenhum arquivo PDF encontrado em data/input/pdfs!")
        return False

    return True


def test_pdf_workflow(pdf_path: Path) -> Optional[Dict[str, Any]]:
    """Test the workflow with a PDF file."""
    print(f"🌳 Teste do Fluxo Pan-Amazônico com PDF: {pdf_path.name}")
    print("=" * 70)

    try:
        # Initialize components
        print("🔧 Inicializando componentes...")
        client = OpenAIClient()
        pm = PanAmazonPromptManager()
        parser = PanAmazonResponseParser()
        pdf_processor = PDFProcessor()

        # Process the PDF file
        print(f"📄 Processando arquivo PDF: {pdf_path.name}...")
        processed_doc = pdf_processor.process_pdf(pdf_path)

        print(f"✅ PDF processado com sucesso!")
        print(f"📊 Idioma detectado: {processed_doc.language['name']} ({processed_doc.language['code']})")
        print(f"📊 Total de páginas: {processed_doc.page_count}")
        print(f"📊 Total de caracteres: {processed_doc.char_count}")
        print(f"📊 Total de chunks: {len(processed_doc.chunks)}")
        print(f"📊 Método de extração: {processed_doc.extraction_method}")

        # Display visual elements information
        if processed_doc.visual_elements:
            print(f"🖼️ Total de elementos visuais: {len(processed_doc.visual_elements)}")

            # Count by type
            type_counts = {}
            for element in processed_doc.visual_elements:
                element_type = element.element_type.value
                type_counts[element_type] = type_counts.get(element_type, 0) + 1

            for element_type, count in type_counts.items():
                print(f"  - {element_type}: {count}")

        # Use the first chunk for analysis (or combine chunks if needed)
        # For simplicity, we'll use just the first chunk in this example
        if len(processed_doc.chunks) > 0:
            analysis_text = processed_doc.chunks[0]
            print(f"📝 Analisando primeiro chunk ({len(analysis_text)} caracteres)...")
        else:
            raise ValueError("Nenhum texto extraído do PDF")

        # Format prompt using client's approach
        source_name = pdf_path.stem
        formatted_prompt = pm.format_extraction_prompt(analysis_text, source_name)
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
            analysis_text
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

        # Prepare visual elements metadata
        visual_elements_metadata = None
        if processed_doc.visual_elements:
            # Count by type
            type_counts = {}
            for element in processed_doc.visual_elements:
                element_type = element.element_type.value
                type_counts[element_type] = type_counts.get(element_type, 0) + 1

            visual_elements_metadata = {
                "total": len(processed_doc.visual_elements),
                "tipos": type_counts,
                # Include a limited number of elements to avoid large files
                "elementos": [
                    {
                        "id": element.id,
                        "tipo": element.element_type.value,
                        "pagina": element.page_number,
                        "largura": element.width,
                        "altura": element.height,
                        "descricao": element.description,
                        "texto_ocr": element.ocr_text
                    } 
                    for element in processed_doc.visual_elements[:10]  # Limit to first 10 elements
                ]
            }

        output_data = {
            "resultado_extracao": result.to_dict(),
            "avaliacao_qualidade": quality,
            "metadados_pdf": {
                "nome_arquivo": pdf_path.name,
                "paginas": processed_doc.page_count,
                "caracteres": processed_doc.char_count,
                "chunks": len(processed_doc.chunks),
                "metodo_extracao": processed_doc.extraction_method,
                "idioma": processed_doc.language,
                "elementos_visuais": visual_elements_metadata
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
        output_file = Path(f"data/output/teste_pdf_{pdf_path.stem}.json")
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"✅ Resultado salvo em: {output_file}")
        except (OSError, IOError) as e:
            print(f"⚠️ Erro ao salvar arquivo: {e}")
            # Continue execution even if file save fails

        print(f"\n🎉 TESTE CONCLUÍDO COM SUCESSO!")
        print(f"Sistema validado com PDF real!")

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
    """Run PDF workflow test."""
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
    result = test_pdf_workflow(sample_pdf)

    if result is not None:
        print("\n✅ Sistema pronto para processar documentos PDF da Pan-Amazônia!")
        return True
    else:
        print("\n❌ Teste falhou. Verifique a configuração.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
