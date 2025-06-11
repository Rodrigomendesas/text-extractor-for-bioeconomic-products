
# Bioeconomic Product Analyzer

A Python application for analyzing PDF documents to extract information about bioeconomic products from both English and Spanish texts, with support for both text extraction and direct file processing approaches.

## Features

- Multiple processing approaches:
  - Text extraction workflow (extracts text from PDFs first)
  - Direct PDF workflow (sends PDFs directly to OpenAI API)
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
   source .venv\Scripts\activate  # On Windows use backslashes
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the root directory and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## For Non-Technical Users: Quick Start Guide

If you're not familiar with programming or command line interfaces, follow these simple steps to run the application:

### Prerequisites
1. Make sure you have Python installed on your computer (version 3.8 or higher)
2. Make sure you have an OpenAI API key

### Running the Application (Windows)
1. Open Command Prompt (search for "cmd" in the Start menu)
2. Navigate to the folder where you saved the application:
   ```
   cd C:\path\to\text-extractor
   ```
3. Activate the virtual environment:
   ```
   .venv\Scripts\activate
   ```
4. Run one of the following commands depending on what you want to do:

   To analyze a single PDF file:
   ```
   python scripts\direct_pdf_workflow.py --file data\input\pdfs\your_document.pdf
   ```

   To process multiple PDF files at once:
   ```
   python scripts\direct_batch_process.py
   ```

5. Find your results in the `data\output` folder:
   - JSON files contain detailed extraction results
   - CSV files contain tabulated product information

### Common Issues
- If you see an error about API keys, check that your `.env` file contains the correct OpenAI API key
- If a file isn't found, double-check the path to make sure it exists

## Usage

### Text Extraction Workflow
This approach extracts text from PDFs first, then sends the text to the OpenAI API for analysis.

#### Single Document Analysis
```
python scripts\run_analysis.py --file data\input\pdfs\document.pdf
```

#### PDF Client Workflow
```
python scripts\pdf_client_workflow.py
```

#### Batch Processing with Text Extraction
```
python scripts\batch_process.py
```

### Direct PDF Workflow
This approach sends PDF files directly to the OpenAI API without extracting text first, simplifying the workflow. This is the recommended approach for most users.

#### Single Document Direct Processing
```
python scripts\direct_pdf_workflow.py --file data\input\pdfs\your_document.pdf
```

#### Batch Processing with Direct Approach
```
python scripts\direct_batch_process.py
```

### Export Results to CSV
After processing documents, you can export the extracted products to a CSV file:

```
python scripts\export_bioeconomy_csv.py
```

### Choosing the Right Approach

#### Text Extraction Workflow
- Provides more control over the extraction process
- Allows for visual element extraction (images, charts, tables)
- Enables preprocessing of text before sending to the API
- Better for complex documents with many visual elements

#### Direct PDF Workflow (Recommended)
- Simpler implementation with fewer steps
- Leverages OpenAI's built-in PDF processing capabilities
- May handle some complex PDFs better (especially scanned documents)
- Reduces code complexity and maintenance
- Potentially more cost-effective for simple documents

### Output Files
All output files are stored in the following locations:
- `data\output\`: Contains JSON files with extraction results and CSV exports
- `logs\`: Contains log files with processing information

Note: These directories are included in the `.gitignore` file and won't be tracked by Git.

### Visual Element Extraction
The system automatically extracts visual elements (images, charts, tables) from PDFs during processing when using the text extraction workflow. These elements are included in the output data and can be used for further analysis.

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

## Project Structure

The project follows a modular structure:
- `src/`: Contains the core application code
  - `core/`: Core functionality (text extraction, PDF processing)
  - `extractors/`: Product extraction logic
  - `llm/`: LLM integration (OpenAI API)
  - `models/`: Data models
  - `storage/`: Database and export functionality
  - `utils/`: Utility functions
- `scripts/`: Contains executable scripts for different workflows
- `data/`: Contains input and output data
  - `input/pdfs/`: Place your PDF files here for analysis
  - `output/`: Contains the extraction results (JSON, CSV)
- `logs/`: Contains log files
- `config/`: Contains configuration files

## Recent Updates

- Added direct PDF processing workflow for simpler operation
- Added batch processing for direct PDF workflow
- Updated .gitignore to exclude:
  - Output files (`data/output/` and `scripts/data/output/`)
  - Log files (`logs/` and `scripts/logs/`)
  - Example files (`data/input/examples/`)
  - Database files (`data/database/*.db`)
  - Error files (`error_detail.txt`)
  - Environment files (`.env`)
- Improved documentation for non-technical users
