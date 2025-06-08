"""
Script to process a specific PDF file and export the results to a CSV file
with the required columns for the bioeconomy marketplace initiative.

Required CSV columns:
- Nome popular do produto
- Nome científico (se disponível)
- País(es) onde é mencionado ou utilizado
- Tipo de uso (alimentar, medicinal, cosmético, artesanal, etc.)
- Fonte (nome do texto)
"""
import sys
import csv
from pathlib import Path
from typing import List, Dict, Any

# Add repository root and src to path
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))
sys.path.append(str(repo_root / "src"))

from config.settings import settings, config_manager
from src.llm.openai_client import OpenAIClient
from src.llm.prompt_manager import PanAmazonPromptManager
from src.llm.response_parser import PanAmazonResponseParser
from src.core.pdf_processor import PDFProcessor
from src.llm.response_parser import SociobiodiversityProduct


def validate_prerequisites() -> bool:
    """Validate all prerequisites before running the script."""
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


def process_pdf(pdf_path: Path) -> List[SociobiodiversityProduct]:
    """Process a PDF file and extract products."""
    print(f"🌳 Processando PDF: {pdf_path.name}")
    print("=" * 70)

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

    # Combine multiple chunks for a more comprehensive analysis
    # We'll use the first 5 chunks or all chunks if there are fewer than 5
    if len(processed_doc.chunks) > 0:
        num_chunks = min(5, len(processed_doc.chunks))
        analysis_text = "\n\n".join(processed_doc.chunks[:num_chunks])
        print(f"📝 Analisando {num_chunks} chunks ({len(analysis_text)} caracteres)...")
    else:
        raise ValueError("Nenhum texto extraído do PDF")

    # Format prompt using client's approach
    source_name = pdf_path.stem
    formatted_prompt = pm.format_extraction_prompt(analysis_text, source_name)

    # Make request with error handling
    print("🤖 Fazendo requisição para OpenAI...")
    response = client.system_user_completion(
        formatted_prompt["system"],
        formatted_prompt["user"]
    )

    # Parse response
    print("🔍 Analisando resposta...")
    result = parser.parse_extraction_response(
        response.content, 
        source_name, 
        analysis_text
    )

    # Display results
    print(f"\n📊 RESULTADOS DA EXTRAÇÃO")
    print(f"Total de produtos encontrados: {len(result.produtos)}")

    return result.produtos


def export_to_csv(products: List[SociobiodiversityProduct], output_path: Path) -> bool:
    """
    Export products to CSV with the required columns.

    Required columns:
    - Nome popular do produto
    - Nome científico (se disponível)
    - País(es) onde é mencionado ou utilizado
    - Tipo de uso (alimentar, medicinal, cosmético, artesanal, etc.)
    - Fonte (nome do texto)
    """
    try:
        # Prepare rows for CSV
        rows = []
        for product in products:
            # Get country
            country = ", ".join(product.paises) if product.paises else ""

            # If product has tipos_uso, create a row for each use
            if product.tipos_uso:
                for uso in product.tipos_uso:
                    row = {
                        'Nome popular do produto': product.nome_popular,
                        'Nome científico (se disponível)': product.nome_cientifico or '',
                        'País(es) onde é mencionado ou utilizado': country,
                        'Tipo de uso (alimentar, medicinal, cosmético, artesanal, etc.)': uso,
                        'Fonte (nome do texto)': product.fonte or ''
                    }
                    rows.append(row)
            else:
                # Product with no uses
                row = {
                    'Nome popular do produto': product.nome_popular,
                    'Nome científico (se disponível)': product.nome_cientifico or '',
                    'País(es) onde é mencionado ou utilizado': country,
                    'Tipo de uso (alimentar, medicinal, cosmético, artesanal, etc.)': '',
                    'Fonte (nome do texto)': product.fonte or ''
                }
                rows.append(row)

        # Write CSV
        if rows:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            print(f"✅ Exportado {len(products)} produtos para CSV: {output_path}")
            return True
        else:
            print("⚠️ Nenhum produto para exportar")
            return False

    except Exception as e:
        print(f"❌ Erro ao exportar para CSV: {e}")
        return False


def main():
    """Run the script to process a PDF and export to CSV."""
    print("Validando configuração...")

    if not validate_prerequisites():
        return False

    # Specify the PDF file to process
    pdf_path = Path("data/input/pdfs/Rainforest Alliance_Bioeconomy-Marketplace-Initiative.pdf")

    if not pdf_path.exists():
        print(f"❌ Arquivo PDF não encontrado: {pdf_path}")
        return False

    print(f"Usando arquivo: {pdf_path.name} ({pdf_path.stat().st_size / 1024:.1f} KB)")

    try:
        # Process the PDF
        products = process_pdf(pdf_path)

        # Export to CSV
        csv_path = Path("data/output/bioeconomy_marketplace_products.csv")
        success = export_to_csv(products, csv_path)

        if success:
            print(f"\n✅ Arquivo CSV gerado com sucesso: {csv_path}")
            return True
        else:
            print("\n❌ Falha ao gerar arquivo CSV.")
            return False

    except Exception as e:
        print(f"❌ Erro durante o processamento: {e}")
        if settings.debug_mode:
            import traceback
            traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
