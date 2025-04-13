"""Tool for reading and parsing CSV files with advanced options."""

import csv
import os
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger
from pydantic import ConfigDict
from quantalogic.tools.tool import Tool, ToolArgument


class CSVReaderTool(Tool):
    """Tool for reading and parsing CSV files with support for various formats and options."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "csv_reader_tool"
    description: str = (
        "Reads and parses CSV files with support for different delimiters, encodings, "
        "and formatting options. Returns data in a structured format."
    )
    need_validation: bool = False

    arguments: List[ToolArgument] = [
        ToolArgument(
            name="file_path",
            arg_type="string",
            description="Path to the CSV file to read",
            required=True,
            example="/path/to/data.csv",
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
            description="File encoding (e.g., utf-8, latin-1)",
            required=False,
            default="utf-8",
            example="utf-8",
        ),
        ToolArgument(
            name="has_headers",
            arg_type="string",
            description="Whether the CSV file has header row",
            required=False,
            default="True",
            example="True",
        ),
        ToolArgument(
            name="max_rows",
            arg_type="string",
            description="Maximum number of rows to read (0 for all)",
            required=False,
            default="0",
            example="1000",
        ),
    ]

    def execute(
        self,
        file_path: str,
        delimiter: str = ",",
        encoding: str = "utf-8",
        has_headers: str = "True",
        max_rows: str = "0",
    ) -> str:
        """Read and parse a CSV file with the specified options.

        Args:
            file_path: Path to the CSV file
            delimiter: CSV delimiter character
            encoding: File encoding
            has_headers: Whether the CSV has headers
            max_rows: Maximum rows to read (0 for all)

        Returns:
            Formatted string representation of the CSV data

        Raises:
            ValueError: If file doesn't exist or parameters are invalid
            Exception: For other errors during reading
        """
        try:
            # Validate file exists
            if not os.path.isfile(file_path):
                raise ValueError(f"File not found: {file_path}")

            # Parse boolean and integer parameters
            has_headers_bool = has_headers.lower() in ["true", "1", "yes"]
            try:
                max_rows_int = int(max_rows)
                if max_rows_int < 0:
                    raise ValueError("max_rows must be >= 0")
            except ValueError as e:
                raise ValueError(f"Invalid max_rows value: {max_rows}") from e

            # Read the CSV file
            data: List[List[str]] = []
            headers: Optional[List[str]] = None

            with open(file_path, "r", encoding=encoding) as f:
                reader = csv.reader(f, delimiter=delimiter)
                
                # Handle headers
                if has_headers_bool:
                    headers = next(reader)
                    
                # Read data rows
                for i, row in enumerate(reader):
                    if max_rows_int > 0 and i >= max_rows_int:
                        break
                    data.append(row)

            # Format the output
            output = []
            
            # Add file info
            output.append(f"File: {file_path}")
            output.append(f"Total rows read: {len(data)}")
            if headers:
                output.append(f"Headers: {', '.join(headers)}")
            output.append("")

            # Add data preview
            preview_rows = min(len(data), 5)  # Show up to 5 rows as preview
            if headers:
                # Calculate column widths
                col_widths = [len(h) for h in headers]
                for row in data[:preview_rows]:
                    for i, val in enumerate(row):
                        col_widths[i] = max(col_widths[i], len(val))

                # Format table with headers
                header_row = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
                output.append(header_row)
                output.append("-" * len(header_row))
                
                # Format data rows
                for row in data[:preview_rows]:
                    output.append(" | ".join(str(val).ljust(w) for val, w in zip(row, col_widths)))
            else:
                # Without headers, just show raw rows
                for row in data[:preview_rows]:
                    output.append(" | ".join(row))

            if len(data) > preview_rows:
                output.append(f"\n... {len(data) - preview_rows} more rows ...")

            return "\n".join(output)

        except UnicodeDecodeError as e:
            logger.error(f"Encoding error reading CSV: {str(e)}")
            raise ValueError(f"Error reading file with encoding {encoding}. Try a different encoding.") from e
        except csv.Error as e:
            logger.error(f"CSV parsing error: {str(e)}")
            raise ValueError(f"Error parsing CSV: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error reading CSV: {str(e)}")
            raise ValueError(f"Failed to read CSV file: {str(e)}") from e


if __name__ == "__main__":
    # Example usage
    tool = CSVReaderTool()
    print(tool.to_markdown())
    
    # Example with a test CSV
    test_csv = "/tmp/test.csv"
    with open(test_csv, "w") as f:
        f.write("Name,Age,City\nJohn,30,New York\nJane,25,London\nBob,35,Paris")
    
    try:
        result = tool.execute(
            file_path=test_csv,
            delimiter=",",
            encoding="utf-8",
            has_headers="True",
            max_rows="5"
        )
        print("\nTest Result:")
        print(result)
    except Exception as e:
        print(f"Error: {e}")
