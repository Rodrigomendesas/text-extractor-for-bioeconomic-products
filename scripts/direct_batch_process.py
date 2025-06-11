"""
Batch processing script for direct document workflow.
Processes multiple PDF, DOC, and DOCX files by sending them directly to OpenAI API.
"""
import sys
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add repository root and src to path
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))
sys.path.append(str(repo_root / "src"))

from config.settings import settings, config_manager
from src.llm.openai_client import OpenAIClient
from src.llm.prompt_manager import PanAmazonPromptManager, PromptType
from src.llm.response_parser import PanAmazonResponseParser
from src.storage.export_manager import ExportManager

# Configure logging
logging.basicConfig(
    level=settings.log_level.value,
    format=settings.log_format,
    handlers=[
        logging.FileHandler(settings.log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def process_file(file_path: Path, client: OpenAIClient, pm: PanAmazonPromptManager, 
                    parser: PanAmazonResponseParser) -> Dict[str, Any]:
    """Process a single file (PDF, DOC, DOCX) using the direct approach."""
    logger.info(f"Processing file: {file_path.name}")

    try:
        # Get the extension
        extension = file_path.suffix.lower()

        # Get the extraction prompt
        extraction_prompt = pm.get_prompt(PromptType.SOCIOBIODIVERSITY_EXTRACTION)

        # Format the prompts
        source_name = file_path.stem
        system_prompt = extraction_prompt.system_prompt

        # Customize message based on file type
        if extension == '.pdf':
            file_type_msg = "PDF"
        elif extension in ['.doc', '.docx']:
            file_type_msg = "documento Word"
        else:
            file_type_msg = "arquivo"

        user_prompt = extraction_prompt.user_template.format(
            text=f"[O conteúdo será extraído diretamente do {file_type_msg}]",
            source_name=source_name
        )

        # Send the file directly to OpenAI API
        response = client.file_completion(
            file_path=str(file_path),
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )

        # Parse response
        result = parser.parse_extraction_response(
            response.content, 
            source_name, 
            ""  # No original text since we're sending the file directly
        )

        # Calculate cost
        cost = settings.estimate_cost(
            response.usage_details["prompt_tokens"],
            response.usage_details["completion_tokens"],
            response.model
        )

        # Quality assessment
        quality = parser.validate_extraction_quality(result)

        # Prepare output data
        output_data = {
            "resultado_extracao": result.to_dict(),
            "avaliacao_qualidade": quality,
            "metadados_arquivo": {
                "nome_arquivo": file_path.name,
                "tipo_arquivo": extension[1:],  # Remove the dot
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

        # Save individual result
        output_file = settings.output_dir / f"direct_{extension[1:]}_{file_path.stem}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Completed processing {file_path.name}: {result.total_produtos} products found")
        return output_data

    except Exception as e:
        logger.error(f"Error processing {file_path.name}: {e}")
        return {
            "error": str(e),
            "file": file_path.name,
            "success": False
        }


def batch_process_files(input_dir: Path, max_workers: int = 2, limit: int = None) -> List[Dict[str, Any]]:
    """Process multiple files (PDF, DOC, DOCX) in parallel using the direct approach."""
    # Initialize components
    client = OpenAIClient()
    pm = PanAmazonPromptManager()
    parser = PanAmazonResponseParser()
    export_manager = ExportManager()

    # Get all supported files
    pdf_files = list(input_dir.glob("*.pdf"))
    doc_files = list(input_dir.glob("*.doc"))
    docx_files = list(input_dir.glob("*.docx"))

    all_files = pdf_files + doc_files + docx_files

    if limit and limit > 0:
        all_files = all_files[:limit]

    if not all_files:
        logger.warning(f"No supported files found in {input_dir}")
        return []

    # Count by type
    file_counts = {
        "pdf": len(pdf_files),
        "doc": len(doc_files),
        "docx": len(docx_files),
        "total": len(all_files)
    }

    logger.info(f"Found {file_counts['total']} files to process: "
               f"{file_counts['pdf']} PDFs, {file_counts['doc']} DOCs, {file_counts['docx']} DOCXs")

    # Sort by size (smallest first)
    all_files.sort(key=lambda p: p.stat().st_size)

    results = []
    start_time = time.time()

    # Process files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(process_file, file, client, pm, parser): file 
            for file in all_files
        }

        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
                logger.info(f"Completed {len(results)}/{len(all_files)}: {file.name}")
            except Exception as e:
                logger.error(f"Exception processing {file.name}: {e}")
                results.append({
                    "error": str(e),
                    "file": file.name,
                    "success": False
                })

    # Calculate statistics
    total_time = time.time() - start_time
    successful = sum(1 for r in results if "error" not in r)
    total_products = sum(r.get("resultado_extracao", {}).get("total_produtos", 0) for r in results if "error" not in r)
    total_cost = sum(r.get("custos", {}).get("custo_usd", 0) for r in results if "error" not in r)

    # Export combined results
    try:
        # Export to JSON
        combined_output = {
            "resultados": results,
            "estatisticas": {
                "total_arquivos": len(all_files),
                "arquivos_processados": len(results),
                "arquivos_com_sucesso": successful,
                "total_produtos": total_products,
                "tempo_total_segundos": total_time,
                "custo_total_usd": total_cost,
                "contagem_por_tipo": file_counts
            }
        }

        output_file = settings.output_dir / "direct_batch_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(combined_output, f, ensure_ascii=False, indent=2)

        # Export products to CSV
        all_products = []
        for result in results:
            if "error" not in result:
                produtos = result.get("resultado_extracao", {}).get("produtos", [])
                all_products.extend(produtos)

        if all_products:
            # Export to CSV
            csv_file = settings.output_dir / "bioeconomy_marketplace_products.csv"
            export_manager.export_products_to_csv(all_products, csv_file)
            logger.info(f"Exported {len(all_products)} products to CSV: {csv_file}")

        logger.info(f"Batch processing complete. Results saved to {output_file}")
        logger.info(f"Processed {successful}/{len(all_files)} files successfully")
        logger.info(f"Found {total_products} products in total")
        logger.info(f"Total cost: ${total_cost:.4f}")
        logger.info(f"Total time: {total_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Error exporting results: {e}")

    return results


def main():
    """Run batch processing for all supported document types."""
    print("Direct Document Batch Processing")
    print("================================")

    # Validate prerequisites
    validation = config_manager.validate_api_keys()
    if not validation["openai"]:
        print("❌ OpenAI API key not configured!")
        return False

    # Ensure output directory exists
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    # Get input directory
    input_dir = settings.input_dir
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return False

    # Get parameters
    max_workers = settings.max_concurrent_requests
    limit = None  # Process all files

    # Count files by type
    pdf_count = len(list(input_dir.glob("*.pdf")))
    doc_count = len(list(input_dir.glob("*.doc")))
    docx_count = len(list(input_dir.glob("*.docx")))
    total_count = pdf_count + doc_count + docx_count

    if total_count == 0:
        print(f"❌ No supported files found in {input_dir}")
        print("Supported formats: PDF, DOC, DOCX")
        return False

    # Ask for confirmation
    print(f"Found {total_count} files to process:")
    print(f"  - {pdf_count} PDF files")
    print(f"  - {doc_count} DOC files")
    print(f"  - {docx_count} DOCX files")

    if limit:
        print(f"Will process up to {limit} of {total_count} files")
    else:
        print(f"Will process all {total_count} files")

    print(f"Using {max_workers} parallel workers")
    print(f"Results will be saved to:")
    print(f"  - JSON: {settings.output_dir / 'direct_batch_results.json'}")
    print(f"  - CSV: {settings.output_dir / 'bioeconomy_marketplace_products.csv'}")

    confirm = input("Continue? (y/n): ")
    if confirm.lower() != 'y':
        print("Operation cancelled")
        return False

    # Process files
    results = batch_process_files(input_dir, max_workers, limit)

    # Print summary
    successful = sum(1 for r in results if "error" not in r)
    total_products = sum(r.get("resultado_extracao", {}).get("total_produtos", 0) for r in results if "error" not in r)

    print("\n✅ Batch processing complete!")
    print(f"✅ Processed {successful}/{len(results)} files successfully")
    print(f"✅ Found {total_products} products in total")
    print(f"✅ Results saved to JSON and CSV files")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
