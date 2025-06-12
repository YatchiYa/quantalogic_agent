# Legal Document Analyzer Flow

A workflow for analyzing legal documents using LLMs with customizable templates. This flow extracts content from various document formats, analyzes the legal content, and generates a comprehensive report.

## Features

- Supports multiple document formats (PDF, Word, Text, Markdown)
- Extracts document metadata (title, type, date, jurisdiction, governing law, parties)
- Identifies key clauses and their importance
- Analyzes potential legal risks and issues
- Provides recommendations based on document content
- Generates a well-structured summary report in Markdown format
- Customizable with user-provided instructions

## Usage

```bash
python legal_document_analyzer.py analyze [OPTIONS] FILE_PATH
```

### Options

- `--text-extraction-model TEXT`: LLM model for PDF/Word text extraction (default: gemini/gemini-2.0-flash)
- `--analysis-model TEXT`: LLM model for document analysis (default: gemini/gemini-2.0-pro)
- `--summary-model TEXT`: LLM model for report generation (default: gemini/gemini-2.0-pro)
- `--output-dir TEXT`: Directory to save output files (supports ~ expansion)
- `--custom-metadata-instructions TEXT`: Custom instructions for metadata extraction
- `--custom-analysis-instructions TEXT`: Custom instructions for document analysis
- `--help`: Show help message and exit

### Example

```bash
python legal_document_analyzer.py analyze ~/contracts/agreement.pdf --analysis-model "openai/gpt-4" --custom-analysis-instructions "Focus on identifying potential liability issues"
```

## Workflow Structure

1. **Document Processing**:
   - Check file type
   - Extract text content based on file type (PDF, Word, Text, Markdown)
   - Save extracted content

2. **Metadata Extraction**:
   - Extract document title, type, date
   - Identify jurisdiction and governing law
   - Identify parties involved and their roles

3. **Document Analysis**:
   - Identify key clauses
   - Assess clause importance (High, Medium, Low)
   - Identify potential issues with each clause
   - Generate comprehensive document summary
   - Identify legal risks
   - Provide recommendations

4. **Report Generation**:
   - Generate a well-structured Markdown report
   - Save report to file

## Dependencies

- pyzerox: For PDF text extraction
- docx2txt: For Word document text extraction
- typer: For CLI interface
- rich: For console output formatting
- loguru: For logging
- pydantic: For data validation and structured output

## Customization

You can customize the analysis by providing specific instructions:

- **Metadata Instructions**: Additional guidance for extracting document metadata
- **Analysis Instructions**: Specific focus areas or requirements for the document analysis

These instructions will be incorporated into the LLM prompts to guide the analysis process.
