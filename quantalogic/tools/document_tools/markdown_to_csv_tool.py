"""Tool for converting markdown content to CSV format.

This tool parses markdown tables and structured content to generate CSV files.
It supports various markdown table formats and can handle multiple tables in a single document.
"""

import csv
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import markdown
from bs4 import BeautifulSoup
from loguru import logger
from quantalogic.tools.tool import Tool, ToolArgument


class MarkdownToCSVTool(Tool):
    """Converts markdown tables and structured content to CSV format."""

    name: str = "markdown_to_csv_tool"
    description: str = (
        "Converts markdown tables and structured content to CSV format. "
        "Supports multiple tables and various markdown formats. "
        "Can extract structured data from markdown lists and headers. "
        "Saves output in /tmp directory."
    )
    need_validation: bool = False

    arguments: List[ToolArgument] = [
        ToolArgument(
            name="markdown_content",
            arg_type="string",
            description="Markdown content containing tables or structured data",
            required=True,
            example='''# Product Catalog

## Electronics
| Product | Price | Stock |
|---------|-------|-------|
| Laptop  | $999  | 50    |
| Phone   | $599  | 100   |
| Tablet  | $299  | 75    |

## Accessories
| Item    | Price | Stock |
|---------|-------|-------|
| Case    | $29   | 200   |
| Charger | $19   | 150   |''',
        ),
        ToolArgument(
            name="output_path",
            arg_type="string",
            description="Path for saving the CSV file(s)",
            required=True,
            example="/tmp/output.csv",
        ),
        ToolArgument(
            name="delimiter",
            arg_type="string",
            description="CSV delimiter character",
            required=False,
            default=",",
            example=",",
        ),
        ToolArgument(
            name="encoding",
            arg_type="string",
            description="Output file encoding",
            required=False,
            default="utf-8",
            example="utf-8",
        ),
        ToolArgument(
            name="extract_lists",
            arg_type="string",
            description="Whether to extract structured data from lists",
            required=False,
            default="False",
            example="True",
        ),
    ]

    def _normalize_path(self, path: str) -> Path:
        """Ensure output path is in /tmp directory.
        
        Args:
            path: Original path
            
        Returns:
            Normalized path in /tmp
        """
        if not path.startswith("/tmp/"):
            filename = os.path.basename(path)
            path = os.path.join("/tmp", filename)
        return Path(path).resolve()

    def _extract_tables(self, html_content: str) -> List[Tuple[str, List[List[str]]]]:
        """Extract tables from HTML content.
        
        Args:
            html_content: HTML string
            
        Returns:
            List of (table title, table data) tuples
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        tables = []
        current_title = "Table"
        
        # Find all tables and their preceding headers
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'table']):
            if element.name != 'table':
                current_title = element.get_text().strip()
            else:
                rows = []
                # Extract headers
                headers = []
                for th in element.find_all('th'):
                    headers.append(th.get_text().strip())
                if headers:
                    rows.append(headers)
                
                # Extract data rows
                for tr in element.find_all('tr'):
                    row = []
                    for td in tr.find_all('td'):
                        row.append(td.get_text().strip())
                    if row:  # Skip empty rows
                        rows.append(row)
                
                if rows:  # Only add if table has content
                    tables.append((current_title, rows))
                
        return tables

    def _extract_lists(self, html_content: str) -> List[Tuple[str, List[List[str]]]]:
        """Extract structured data from lists.
        
        Args:
            html_content: HTML string
            
        Returns:
            List of (list title, structured data) tuples
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        lists = []
        current_title = "List"
        
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol']):
            if element.name not in ['ul', 'ol']:
                current_title = element.get_text().strip()
            else:
                rows = []
                headers = ["Item", "Details"]
                rows.append(headers)
                
                for li in element.find_all('li'):
                    text = li.get_text().strip()
                    # Try to split into key-value if possible
                    if ':' in text:
                        key, value = text.split(':', 1)
                        rows.append([key.strip(), value.strip()])
                    else:
                        rows.append([text, ""])
                
                if len(rows) > 1:  # Only add if list has content
                    lists.append((current_title, rows))
        
        return lists

    def execute(
        self,
        markdown_content: str,
        output_path: str,
        delimiter: str = ",",
        encoding: str = "utf-8",
        extract_lists: str = "False",
    ) -> str:
        """Convert markdown content to CSV format.

        Args:
            markdown_content: Input markdown text
            output_path: Where to save the CSV file(s)
            delimiter: CSV delimiter character
            encoding: Output file encoding
            extract_lists: Whether to extract data from lists

        Returns:
            Status message with output file information

        Raises:
            ValueError: If parameters are invalid
            Exception: For other conversion errors
        """
        try:
            # Convert extract_lists to boolean
            extract_lists_bool = extract_lists.lower() in ["true", "1", "yes"]
            
            # Convert markdown to HTML
            html = markdown.markdown(markdown_content, extensions=['tables'])
            
            # Extract tables and optionally lists
            all_data = self._extract_tables(html)
            if extract_lists_bool:
                all_data.extend(self._extract_lists(html))
            
            if not all_data:
                raise ValueError("No tables or structured data found in markdown content")
            
            # Prepare output paths
            base_path = self._normalize_path(output_path)
            base_name = base_path.stem
            extension = base_path.suffix
            
            # Write each table to a separate CSV file
            output_files = []
            for i, (title, rows) in enumerate(all_data):
                # Create filename
                if len(all_data) > 1:
                    # If multiple tables, add index to filename
                    file_path = base_path.parent / f"{base_name}_{i+1}{extension}"
                else:
                    file_path = base_path
                
                # Ensure parent directory exists
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write CSV file
                with open(file_path, 'w', newline='', encoding=encoding) as f:
                    writer = csv.writer(f, delimiter=delimiter)
                    writer.writerows(rows)
                
                output_files.append((title, file_path, len(rows)))
            
            # Generate status message
            output = ["Conversion completed successfully:"]
            for title, path, row_count in output_files:
                output.append(f"\n{title}:")
                output.append(f"- File: {path}")
                output.append(f"- Rows: {row_count}")
            
            return "\n".join(output)

        except Exception as e:
            logger.error(f"Error converting markdown to CSV: {str(e)}")
            raise ValueError(f"Failed to convert markdown to CSV: {str(e)}") from e


if __name__ == "__main__":
    # Example usage
    tool = MarkdownToCSVTool()
    print(tool.to_markdown())
    
    # Test with sample markdown
    test_markdown = '''# Product Catalog

## Electronics
| Product | Price | Stock |
|---------|-------|-------|
| Laptop  | $999  | 50    |
| Phone   | $599  | 100   |

## Features
- Basic: Standard features
- Premium: Advanced features
- Pro: All features included'''
    
    try:
        result = tool.execute(
            markdown_content=test_markdown,
            output_path="/tmp/test_output.csv",
            extract_lists="True"
        )
        print("\nTest Result:")
        print(result)
    except Exception as e:
        print(f"Error: {e}")
