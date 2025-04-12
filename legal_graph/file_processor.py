import os
from typing import List, Dict, Optional
from pathlib import Path
import magic
import markdown
from PyPDF2 import PdfReader
from loguru import logger
from document_processor import DocumentProcessor

class FileProcessor:
    def __init__(self, doc_processor: DocumentProcessor):
        self.doc_processor = doc_processor
        self.mime = magic.Magic(mime=True)

    def _read_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {e}")
            return ""

    def _read_markdown(self, file_path: str) -> str:
        """Read and convert markdown to text."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
            # Convert markdown to HTML and strip tags for plain text
            html = markdown.markdown(md_content)
            # Simple HTML tag stripping (you might want to use BeautifulSoup for better results)
            text = html.replace('<p>', '\n').replace('</p>', '\n')
            for tag in ['<h1>', '</h1>', '<h2>', '</h2>', '<h3>', '</h3>', '<strong>', '</strong>', '<em>', '</em>']:
                text = text.replace(tag, '')
            return text
        except Exception as e:
            logger.error(f"Error reading Markdown {file_path}: {e}")
            return ""

    def _extract_metadata(self, file_path: str) -> Dict:
        """Extract basic metadata from filename and path."""
        path = Path(file_path)
        # Try to extract document type and number from filename
        # Example filename: loi_21-45_protection_donnees.pdf
        parts = path.stem.split('_')
        doc_type = parts[0] if len(parts) > 0 else "unknown"
        number = parts[1] if len(parts) > 1 else "unknown"
        title = ' '.join(parts[2:]).replace('_', ' ') if len(parts) > 2 else path.stem
        
        return {
            "type": doc_type,
            "number": number,
            "title": title,
            "file_path": str(path),
            "file_type": path.suffix[1:]  # Remove the dot from extension
        }

    def process_file(self, file_path: str) -> Optional[Dict]:
        """Process a single file and return extracted information."""
        try:
            mime_type = self.mime.from_file(file_path)
            metadata = self._extract_metadata(file_path)
            
            # Extract text based on file type
            if mime_type == 'application/pdf':
                text = self._read_pdf(file_path)
            elif mime_type in ['text/markdown', 'text/plain']:
                text = self._read_markdown(file_path)
            else:
                logger.warning(f"Unsupported file type: {mime_type} for {file_path}")
                return None

            # Process the document
            doc, relations = self.doc_processor.process_document(
                text=text,
                doc_type=metadata['type'],
                number=metadata['number'],
                title=metadata['title']
            )

            return {
                "document": doc,
                "relations": relations,
                "metadata": metadata,
                "text": text
            }

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return None

    def process_directory(self, directory_path: str) -> List[Dict]:
        """Process all PDF and Markdown files in a directory."""
        results = []
        directory = Path(directory_path)
        
        if not directory.exists():
            logger.error(f"Directory not found: {directory_path}")
            return results

        for file_path in directory.rglob('*'):
            if file_path.suffix.lower() in ['.pdf', '.md', '.markdown']:
                logger.info(f"Processing file: {file_path}")
                result = self.process_file(str(file_path))
                if result:
                    results.append(result)

        return results
