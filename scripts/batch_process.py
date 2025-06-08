"""
Script to process all PDF files in the input directory and export the aggregated results 
to a single CSV file with the required columns for the bioeconomy marketplace initiative.

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
import logging
import time

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path("logs/bioeconomic_analysis.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def validate_prerequisites() -> bool:
    """Validate all prerequisites before running the script."""
    # Check API key
    validation = config_manager.validate_api_keys()
    if not validation["openai"]:
        logger.error("❌ Chave da API OpenAI não configurada!")
        return False

    # Ensure output directory exists
    output_dir = Path("data/output")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"❌ Erro ao criar diretório de saída: {e}")
        return False

    # Check if PDF files exist
    pdf_dir = Path("data/input/pdfs")
    if not pdf_dir.exists() or not list(pdf_dir.glob("*.pdf")):
        logger.error("❌ Nenhum arquivo PDF encontrado em data/input/pdfs!")
        return False

    return True


def process_pdf(pdf_path: Path) -> List[SociobiodiversityProduct]:
    """Process a PDF file and extract products."""
    logger.info(f"🌳 Processando PDF: {pdf_path.name}")
    logger.info("=" * 70)

    try:
        # Initialize components
        logger.info("🔧 Inicializando componentes...")
        client = OpenAIClient()
        pm = PanAmazonPromptManager()
        parser = PanAmazonResponseParser()
        pdf_processor = PDFProcessor()

        # Process the PDF file
        logger.info(f"📄 Processando arquivo PDF: {pdf_path.name}...")
        processed_doc = pdf_processor.process_pdf(pdf_path)

        logger.info(f"✅ PDF processado com sucesso!")
        logger.info(f"📊 Idioma detectado: {processed_doc.language['name']} ({processed_doc.language['code']})")
        logger.info(f"📊 Total de páginas: {processed_doc.page_count}")
        logger.info(f"📊 Total de caracteres: {processed_doc.char_count}")
        logger.info(f"📊 Total de chunks: {len(processed_doc.chunks)}")

        # Combine multiple chunks for a more comprehensive analysis
        # We'll use the first 5 chunks or all chunks if there are fewer than 5
        if len(processed_doc.chunks) > 0:
            num_chunks = min(5, len(processed_doc.chunks))
            analysis_text = "\n\n".join(processed_doc.chunks[:num_chunks])
            logger.info(f"📝 Analisando {num_chunks} chunks ({len(analysis_text)} caracteres)...")
        else:
            raise ValueError("Nenhum texto extraído do PDF")

        # Format prompt using client's approach
        source_name = pdf_path.stem
        formatted_prompt = pm.format_extraction_prompt(analysis_text, source_name)

        # Make request with error handling
        logger.info("🤖 Fazendo requisição para OpenAI...")
        response = client.system_user_completion(
            formatted_prompt["system"],
            formatted_prompt["user"]
        )

        # Parse response
        logger.info("🔍 Analisando resposta...")
        result = parser.parse_extraction_response(
            response.content, 
            source_name, 
            analysis_text
        )

        # Display results
        logger.info(f"\n📊 RESULTADOS DA EXTRAÇÃO")
        logger.info(f"Total de produtos encontrados: {len(result.produtos)}")

        return result.produtos

    except Exception as e:
        logger.error(f"❌ Erro ao processar PDF {pdf_path.name}: {e}")
        if settings.debug_mode:
            import traceback
            logger.error(traceback.format_exc())
        return []


def export_to_csv(all_products: List[SociobiodiversityProduct], output_path: Path) -> bool:
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
        for product in all_products:
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

            logger.info(f"✅ Exportado {len(all_products)} produtos para CSV: {output_path}")
            return True
        else:
            logger.warning("⚠️ Nenhum produto para exportar")
            return False

    except Exception as e:
        logger.error(f"❌ Erro ao exportar para CSV: {e}")
        if settings.debug_mode:
            import traceback
            logger.error(traceback.format_exc())
        return False


def process_all_pdfs() -> List[SociobiodiversityProduct]:
    """Process all PDF files in the input directory and return all products."""
    pdf_dir = Path("data/input/pdfs")
    pdf_files = list(pdf_dir.glob("*.pdf"))

    logger.info(f"Encontrados {len(pdf_files)} arquivos PDF para processar")

    all_products = []

    for i, pdf_path in enumerate(pdf_files, 1):
        logger.info(f"Processando arquivo {i}/{len(pdf_files)}: {pdf_path.name}")

        # Add a small delay between files to avoid rate limiting
        if i > 1:
            time.sleep(2)

        # Process the PDF and get products
        products = process_pdf(pdf_path)

        # Add to the aggregated list
        all_products.extend(products)

        logger.info(f"Extraídos {len(products)} produtos de {pdf_path.name}")
        logger.info(f"Total de produtos até agora: {len(all_products)}")

    return all_products


def main():
    """Run the script to process all PDFs and export to a single CSV."""
    logger.info("Iniciando processamento em lote de documentos PDF")
    logger.info("Validando configuração...")

    if not validate_prerequisites():
        return False

    try:
        # Process all PDFs
        start_time = time.time()
        all_products = process_all_pdfs()
        processing_time = time.time() - start_time

        logger.info(f"Processamento concluído em {processing_time:.2f} segundos")
        logger.info(f"Total de produtos extraídos: {len(all_products)}")

        # Export to CSV
        csv_path = Path("data/output/bioeconomy_marketplace_products.csv")
        success = export_to_csv(all_products, csv_path)

        if success:
            logger.info(f"\n✅ Arquivo CSV agregado gerado com sucesso: {csv_path}")
            return True
        else:
            logger.error("\n❌ Falha ao gerar arquivo CSV agregado.")
            return False

    except Exception as e:
        logger.error(f"❌ Erro durante o processamento em lote: {e}")
        if settings.debug_mode:
            import traceback
            logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
