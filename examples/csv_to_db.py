#!/usr/bin/env python3
"""
CSV to Database Loader
This script loads data from a CSV file, automatically detects the schema,
and inserts the data into a PostgreSQL database.
"""

import argparse
import csv
import os
import sys
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from sqlalchemy import create_engine, Table, Column, MetaData, text
from sqlalchemy.types import Integer, Float, String, Boolean, Date, DateTime, Text
import logging
from loguru import logger

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def detect_column_type(values: List[Any]) -> Tuple[Any, int]:
    """
    Detect the data type of a column based on its values.
    Returns the SQLAlchemy type and the max length for string columns.
    """
    # Remove None/empty values for type detection
    non_empty_values = [v for v in values if v is not None and v != '']
    
    if not non_empty_values:
        return String, 255  # Default to string if no values
    
    # Check for percentage values (e.g., '50%')
    percent_values = [str(v).endswith('%') for v in non_empty_values]
    if all(percent_values):
        return String, 10  # Store percentages as strings
    
    # Try to convert to different types
    # Check if all values are integers
    try:
        [int(v) for v in non_empty_values]
        return Integer, 0
    except:
        pass
    
    # Check if all values are floats
    try:
        [float(v) for v in non_empty_values]
        return Float, 0
    except:
        pass
    
    # Check if all values are dates
    try:
        pd.to_datetime(non_empty_values)
        return DateTime, 0
    except:
        pass
    
    # Check if all values are boolean-like (do this last to avoid misclassifying numbers)
    bool_like_values = ['true', 'false', 'yes', 'no', 't', 'f', 'y', 'n']
    if all(str(v).lower() in bool_like_values for v in non_empty_values):
        return Boolean, 0
    
    # If all else fails, it's a string
    # Calculate max length for the string column
    max_length = max([len(str(v)) for v in non_empty_values], default=255)
    
    # Use Text type for very long strings
    if max_length > 5000:
        return Text, 0
    
    # Add some buffer and round to nearest 100 for regular strings
    max_length = 5000
    
    return String, max_length

def create_table_from_csv(
    engine: Any, 
    csv_file: str, 
    table_name: str, 
    delimiter: str = ',', 
    quotechar: str = '"',
    sample_size: int = 1000,
    schema: Optional[str] = None
) -> Table:
    """
    Create a database table based on the schema detected from a CSV file.
    
    Args:
        engine: SQLAlchemy engine
        csv_file: Path to the CSV file
        table_name: Name for the new table
        delimiter: CSV delimiter character
        quotechar: CSV quote character
        sample_size: Number of rows to sample for type detection
        schema: Database schema name (optional)
        
    Returns:
        The created SQLAlchemy Table object
    """
    metadata = MetaData()
    
    # Read CSV headers and sample data
    with open(csv_file, 'r', newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
        headers = next(reader)
        
        # Sample rows for type detection
        sample_data: Dict[str, List[Any]] = {header: [] for header in headers}
        for i, row in enumerate(reader):
            if i >= sample_size:
                break
            for j, value in enumerate(row):
                if j < len(headers):  # Ensure we don't go out of bounds
                    sample_data[headers[j]].append(value)
    
    # Create columns with detected types
    columns = []
    for header in headers:
        # For safety, use Text type for all columns that might contain large text
        if any(len(str(v)) > 1000 for v in sample_data[header] if v):
            columns.append(Column(header, Text))
            logger.info(f"Using Text type for column '{header}' due to large content")
        else:
            col_type, max_length = detect_column_type(sample_data[header])
            
            # Create the column with the appropriate type
            if col_type == String:
                columns.append(Column(header, col_type(max_length)))
            else:
                columns.append(Column(header, col_type))
    
    # Create the table
    table = Table(table_name, metadata, *columns, schema=schema)
    
    # Create the table in the database
    metadata.create_all(engine)
    logger.info(f"Created table '{table_name}' with {len(columns)} columns")
    
    return table

def load_csv_to_db(
    db_url: str,
    csv_file: str,
    table_name: str,
    delimiter: str = ',',
    quotechar: str = '"',
    batch_size: int = 1000,  # Reduced batch size for large text fields
    if_exists: str = 'replace',
    schema: Optional[str] = None
) -> int:
    """
    Load CSV data into a database table.
    
    Args:
        db_url: Database connection URL
        csv_file: Path to the CSV file
        table_name: Name for the table
        delimiter: CSV delimiter character
        quotechar: CSV quote character
        batch_size: Number of rows to insert in each batch
        if_exists: What to do if the table exists ('fail', 'replace', or 'append')
        schema: Database schema name (optional)
        
    Returns:
        Number of rows inserted
    """
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    logger.info(f"Connecting to database: {db_url.replace('://', '://*****@')}")
    engine = create_engine(db_url)
    
    # If replacing, drop the table if it exists
    if if_exists == 'replace':
        with engine.connect() as conn:
            if schema:
                conn.execute(text(f"DROP TABLE IF EXISTS {schema}.{table_name}"))
            else:
                conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
        logger.info(f"Dropped existing table '{table_name}'")
    
    # Create the table based on CSV schema
    table = create_table_from_csv(
        engine, 
        csv_file, 
        table_name, 
        delimiter, 
        quotechar,
        schema=schema
    )
    
    # Use pandas for more robust CSV loading
    try:
        # First try pandas which handles many edge cases better
        df = pd.read_csv(csv_file, delimiter=delimiter, quotechar=quotechar, encoding='utf-8-sig')
        
        # Insert data in smaller batches to handle large text fields
        total_rows = len(df)
        for i in range(0, total_rows, batch_size):
            batch_df = df.iloc[i:i+batch_size]
            records = batch_df.to_dict('records')
            
            try:
                with engine.begin() as conn:
                    conn.execute(table.insert(), records)
                rows_inserted = min(i + batch_size, total_rows)
                logger.info(f"Inserted {rows_inserted} of {total_rows} rows so far")
            except Exception as e:
                logger.error(f"Error inserting batch starting at row {i}: {str(e)}")
                # If pandas approach fails, fall back to manual CSV processing
                raise
        
        logger.info(f"Successfully loaded {total_rows} rows into table '{table_name}'")
        return total_rows
        
    except Exception as e:
        logger.warning(f"Pandas approach failed: {str(e)}. Falling back to manual CSV processing.")
        
        # Fall back to manual CSV processing if pandas approach fails
        rows_inserted = 0
        with open(csv_file, 'r', newline='', encoding='utf-8-sig') as f:
            reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
            headers = next(reader)
            
            batch = []
            for row in reader:
                if len(row) == len(headers):  # Skip malformed rows
                    record = {headers[i]: row[i] for i in range(len(headers))}
                    batch.append(record)
                    
                    if len(batch) >= batch_size:
                        try:
                            with engine.begin() as conn:
                                conn.execute(table.insert(), batch)
                            rows_inserted += len(batch)
                            logger.info(f"Inserted {rows_inserted} rows so far")
                        except Exception as batch_error:
                            logger.error(f"Error inserting batch: {str(batch_error)}")
                            # Try inserting one by one to identify problematic records
                            for idx, record in enumerate(batch):
                                try:
                                    with engine.begin() as conn:
                                        conn.execute(table.insert(), [record])
                                    rows_inserted += 1
                                except Exception as record_error:
                                    logger.error(f"Error inserting record {idx} in batch: {str(record_error)}")
                        batch = []
            
            # Insert any remaining rows
            if batch:
                try:
                    with engine.begin() as conn:
                        conn.execute(table.insert(), batch)
                    rows_inserted += len(batch)
                except Exception as e:
                    logger.error(f"Error inserting final batch: {str(e)}")
                    # Try inserting one by one
                    for idx, record in enumerate(batch):
                        try:
                            with engine.begin() as conn:
                                conn.execute(table.insert(), [record])
                            rows_inserted += 1
                        except Exception as record_error:
                            logger.error(f"Error inserting record {idx} in final batch: {str(record_error)}")
        
        logger.info(f"Successfully loaded {rows_inserted} rows into table '{table_name}'")
        return rows_inserted

def main():
    parser = argparse.ArgumentParser(description='Load CSV data into a PostgreSQL database.')
    parser.add_argument('csv_file', help='Path to the CSV file')
    parser.add_argument('--table', '-t', required=True, help='Name for the database table')
    parser.add_argument('--db-url', '-d', default=os.environ.get('DATABASE_URL', 'postgresql://quantadbu:azerty1234@db:5432/quanta_db'),
                        help='Database connection URL')
    parser.add_argument('--delimiter', default=',', help='CSV delimiter character')
    parser.add_argument('--quotechar', default='"', help='CSV quote character')
    parser.add_argument('--batch-size', type=int, default=10000, help='Number of rows to insert in each batch')
    parser.add_argument('--if-exists', choices=['fail', 'replace', 'append'], default='replace',
                        help='What to do if the table exists')
    parser.add_argument('--schema', help='Database schema name (optional)')
    
    args = parser.parse_args()
    
    try:
        rows = load_csv_to_db(
            args.db_url,
            args.csv_file,
            args.table,
            args.delimiter,
            args.quotechar,
            args.batch_size,
            args.if_exists,
            args.schema
        )
        logger.info(f"CSV import completed successfully. {rows} rows imported.")
        return 0
    except Exception as e:
        logger.error(f"Error importing CSV: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
