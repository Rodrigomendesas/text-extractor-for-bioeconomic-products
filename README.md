
# Bioeconomic Product Analyzer

A Python application for analyzing PDF documents to extract information about bioeconomic products from both English and Spanish texts, including the ability to process visual elements such as images, charts, and tables.

## Features

- PDF text extraction and processing
- Visual element extraction (images, charts, tables)
- Multi-language support (English/Spanish)
- LLM-powered product identification
- Structured data extraction (product name, scientific name, country, uses, etc.)
- Visual content analysis and integration
- Batch processing capabilities
- Export to multiple formats (JSON, CSV, Excel)

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and configure your API keys

## Usage

### Single Document Analysis
```bash
python scripts/run_analysis.py --file data/input/pdfs/document.pdf
```

### Visual Element Extraction
The system automatically extracts visual elements (images, charts, tables) from PDFs during processing. These elements are included in the output data and can be used for further analysis.

```bash
python scripts/pdf_client_workflow.py
```

## Visual Element Processing

The system can extract and process the following types of visual elements:

- **Images**: Extracts embedded images from PDFs with position and metadata
- **Tables**: Detects and extracts tables as images for further processing
- **Charts/Diagrams**: Identifies and extracts charts and diagrams

Visual elements are stored with the following information:
- Element type (image, chart, table, etc.)
- Page number
- Position on page
- Dimensions (width, height)
- Content (base64 encoded)
- Optional OCR text and descriptions
