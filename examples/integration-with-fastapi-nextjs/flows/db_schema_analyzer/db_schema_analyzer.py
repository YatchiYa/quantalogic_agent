"""Database schema analyzer and sample data generator."""

import os
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import click
from loguru import logger

from quantalogic.flow.flow import Nodes, Workflow, WorkflowEvent, WorkflowEventType
from quantalogic.tools.sql_query_tool import SQLQueryTool
from sqlalchemy import MetaData, inspect, text
from sqlalchemy.ext.asyncio import create_async_engine
import asyncio
from typing import Any, Dict, List, Optional
import os
from datetime import datetime

import anyio
from loguru import logger
from pydantic import BaseModel, Field

from quantalogic.flow.flow import Nodes, Workflow, WorkflowEvent, WorkflowEventType
from quantalogic.tools.sql_query_tool import SQLQueryTool
from sqlalchemy import create_engine, MetaData, inspect, text
import asyncio
from typing import Any, Dict, List, Optional
import os
from datetime import datetime

import anyio
from loguru import logger
from pydantic import BaseModel, Field

from quantalogic.flow.flow import Nodes, Workflow, WorkflowEvent, WorkflowEventType
from quantalogic.tools.sql_query_tool import SQLQueryTool
from sqlalchemy import create_engine, MetaData, inspect, text
from sqlalchemy.engine import URL
from sqlalchemy.schema import Table

# Configure logging
logger.remove()
logger.add(
    sink=lambda msg: print(msg, end=""),
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

# Define structured output models
class ColumnInfo(BaseModel):
    name: str
    type: str
    nullable: bool
    primary_key: bool
    foreign_key: Optional[str] = None
    default: Optional[str] = None

class TableSchema(BaseModel):
    name: str
    columns: List[ColumnInfo]
    relationships: List[str] = Field(default_factory=list)
    sample_count: int = 10

class DatabaseSchema(BaseModel):
    tables: List[TableSchema]
    source_db_url: str
    target_db_url: str
    instructions: str = Field(
        default="",
        description="Natural language instructions for data generation"
    )

# Get the templates directory path
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

def get_template_path(template_name):
    return os.path.join(TEMPLATES_DIR, template_name)

# Custom Observer for Workflow Events
async def schema_analyzer_observer(event: WorkflowEvent):
    if event.event_type == WorkflowEventType.WORKFLOW_STARTED:
        print(f"\n{'='*50}\n🚀 Starting Database Schema Analysis 🚀\n{'='*50}")
    elif event.event_type == WorkflowEventType.NODE_STARTED:
        print(f"\n🔄 [{event.node_name}] Starting...")
    elif event.event_type == WorkflowEventType.NODE_COMPLETED:
        print(f"✅ [{event.node_name}] Completed")
    elif event.event_type == WorkflowEventType.WORKFLOW_COMPLETED:
        print(f"\n{'='*50}\n🎉 Database Analysis and Sample Data Generation Complete 🎉\n{'='*50}")

# Workflow Nodes
@Nodes.define(output=None)
async def connect_source_database(db_url: str) -> dict:
    """Connect to the source database and return the SQL query tool."""
    try:
        sql_tool = SQLQueryTool(connection_string=db_url)
        # Test connection by running a simple query
        sql_tool.execute("SELECT 1", 1, 1)
        logger.info(f"Successfully connected to source database")
        return {"sql_tool": sql_tool}
    except Exception as e:
        logger.error(f"Failed to connect to source database: {str(e)}")
        raise

@Nodes.define(output=None)
async def analyze_database_schema(sql_tool: SQLQueryTool, specific_tables: Optional[List[str]] = None) -> dict:
    """Analyze the database schema using SQL queries."""
    try:
        # Get list of tables
        tables_query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        """
        tables_result = sql_tool.execute(tables_query, 1, 100)
        
        # Convert markdown table to list of dicts
        import re
        # Skip header and separator lines
        table_lines = tables_result.strip().split('\n')[2:]
        table_names = [re.sub(r'\s*\|\s*', '', line).strip() for line in table_lines if line.strip()]
        
        tables = []
        for table_name in table_names:
            if specific_tables and table_name not in specific_tables:
                continue
            
            # Get column information
            columns_query = f"""
            SELECT 
                column_name,
                data_type,
                is_nullable,
                column_default,
                (SELECT true 
                 FROM information_schema.key_column_usage 
                 WHERE table_name = c.table_name 
                 AND column_name = c.column_name 
                 LIMIT 1) as is_key
            FROM information_schema.columns c
            WHERE table_name = '{table_name}'
            """
            columns_result = sql_tool.execute(columns_query, 1, 100)
            
            # Parse markdown table
            column_lines = columns_result.strip().split('\n')[2:]  # Skip header and separator
            columns = []
            
            for line in column_lines:
                if not line.strip():
                    continue
                # Split by | and clean up whitespace
                parts = [p.strip() for p in line.split('|')[1:-1]]  # Remove empty first/last after split
                if len(parts) != 5:  # Skip malformed lines
                    continue
                    
                col_name, data_type, is_nullable, col_default, is_key = parts
                
                # Get foreign key information
                fk_query = f"""
                SELECT
                    kcu.column_name,
                    ccu.table_name AS foreign_table_name,
                    ccu.column_name AS foreign_column_name
                FROM 
                    information_schema.table_constraints AS tc 
                    JOIN information_schema.key_column_usage AS kcu
                    ON tc.constraint_name = kcu.constraint_name
                    JOIN information_schema.constraint_column_usage AS ccu
                    ON ccu.constraint_name = tc.constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY' 
                AND tc.table_name = '{table_name}'
                AND kcu.column_name = '{col_name}'
                """
                fk_result = sql_tool.execute(fk_query, 1, 1)
                
                # Parse foreign key result
                fk_info = None
                if fk_result and len(fk_result.strip().split('\n')) > 2:  # Has results
                    fk_line = fk_result.strip().split('\n')[2]  # Skip header and separator
                    if fk_line.strip():
                        fk_parts = [p.strip() for p in fk_line.split('|')[1:-1]]
                        if len(fk_parts) == 3:
                            _, foreign_table, foreign_column = fk_parts
                            fk_info = f"{foreign_table}.{foreign_column}"
                
                columns.append(ColumnInfo(
                    name=col_name,
                    type=data_type,
                    nullable=is_nullable.lower() == 'yes',
                    primary_key=is_key.lower() == 'true',
                    foreign_key=fk_info,
                    default=col_default if col_default != 'None' else None
                ))
            
            # Get relationships
            relationships_query = f"""
            SELECT
                tc.constraint_name,
                kcu.column_name,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name
            FROM 
                information_schema.table_constraints AS tc 
                JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.constraint_column_usage AS ccu
                ON ccu.constraint_name = tc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY' 
            AND tc.table_name = '{table_name}'
            """
            relationships_result = sql_tool.execute(relationships_query, 1, 100)
            
            # Parse relationships result
            relationships = []
            rel_lines = relationships_result.strip().split('\n')[2:]  # Skip header and separator
            for line in rel_lines:
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split('|')[1:-1]]
                if len(parts) == 4:
                    _, column, foreign_table, foreign_column = parts
                    relationships.append(
                        f"References {foreign_table} ({foreign_column}) via {column}"
                    )
            
            tables.append(TableSchema(
                name=table_name,
                columns=columns,
                relationships=relationships
            ))
    
        return {
            "schema": DatabaseSchema(
                tables=tables,
                source_db_url="",  # Will be filled in by the workflow
                target_db_url=""   # Will be filled in by the workflow
            )
        }
    except Exception as e:
        logger.error(f"Failed to analyze database schema: {str(e)}")
        raise

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_generate_sample_data.j2"),
    output="sample_data",
    prompt_file=get_template_path("prompt_generate_sample_data.j2"),
    temperature=0.7,
)
async def generate_sample_data(engine: Any, model: str, schema: DatabaseSchema, instructions: str = "") -> Dict[str, List[Dict[str, Any]]]:
    """Generate sample data for a table based on its schema."""
    if not schema or not schema.tables:
        raise ValueError("No tables found in schema")
    
    table_schema = schema.tables[0]  # For now, just use the first table
    logger.debug(f"Generating sample data for table: {table_schema.name}")
    logger.debug(f"Using instructions: {instructions}")
    return {}

@Nodes.define(output=None)
async def connect_target_database(db_url: str) -> dict:
    """Connect to the target database."""
    try:
        engine = create_engine(db_url)
        # Test connection
        with engine.connect() as conn:
            pass
        logger.info(f"Successfully connected to target database")
        return {"target_engine": engine}
    except Exception as e:
        logger.error(f"Failed to connect to target database: {str(e)}")
        raise

@Nodes.define(output=None)
async def create_target_schema(target_engine: Any, schema: DatabaseSchema) -> None:
    """Create the schema in the target database."""
    metadata = MetaData()
    
    for table_schema in schema.tables:
        # Create SQLAlchemy Table object
        Table(table_schema.name, metadata)
        
    metadata.create_all(target_engine)
    logger.info("Created target database schema")

@Nodes.define(output=None)
async def save_sample_data(target_engine: Any, sample_data: Dict[str, List[Dict[str, Any]]]) -> None:
    """Save the generated sample data using SQL queries."""
    try:
        sql_tool = SQLQueryTool(connection_string=str(target_engine.url))
        
        for table_name, records in sample_data.items():
            if records:
                # Convert records to SQL INSERT statements
                columns = records[0].keys()
                values_list = []
                for record in records:
                    values = [f"'{str(v)}'" if v is not None else 'NULL' for v in record.values()]
                    values_list.append(f"({', '.join(values)})")
                
                insert_query = f"""
                INSERT INTO {table_name} 
                ({', '.join(columns)})
                VALUES {', '.join(values_list)}
                """
                
                sql_tool.execute(insert_query, 1, 1)
                
        logger.info("Successfully saved sample data to target database")
    except Exception as e:
        logger.error(f"Failed to save sample data: {str(e)}")
        raise

# Define the Workflow
workflow = (
    Workflow("connect_source_database")
    .add_observer(schema_analyzer_observer)
    .then("analyze_database_schema")
    .then("generate_sample_data", lambda ctx: {
        "engine": ctx.get("engine"),
        "model": ctx.get("model"),
        "schema": ctx["schema"],
        "instructions": ctx.get("instructions", "")
    })
    .then("connect_target_database")
    .then("create_target_schema")
    .then("save_sample_data")
)

def analyze_and_generate_samples(
    source_db_url: str,
    target_db_url: str,
    model: str = "gemini/gemini-2.0-flash",
    task_id: str = "default",
    specific_tables: Optional[List[str]] = None,
    instructions: Optional[str] = None,
    table_instructions: Optional[Dict[str, str]] = None
) -> None:
    """
    Analyze a database schema and generate sample data.
    
    Args:
        source_db_url: URL of the source database to analyze
        target_db_url: URL of the target database to create and populate
        model: LLM model to use for sample data generation
        task_id: Optional task identifier
        specific_tables: Optional list of table names to analyze. If None, analyzes all tables.
        instructions: Optional natural language instructions for data generation
            Example: "Generate data for a healthcare system. Users should be medical staff with roles 
                     like doctors and nurses. Appointments should be during business hours."
        table_instructions: Optional specific instructions for each table
            Example: {"users": "Include admin roles and medical specialties", 
                     "appointments": "Make sure to include some emergency appointments"}
    """
    initial_context = {
        "db_url": source_db_url,
        "target_db_url": target_db_url,
        "model": model,
        "specific_tables": specific_tables,
        "instructions": instructions or "",
        "table_instructions": table_instructions or {}
    }

    logger.info(f"Starting database analysis for {source_db_url}")
    if specific_tables:
        logger.info(f"Analyzing specific tables: {', '.join(specific_tables)}")
    if instructions:
        logger.info(f"Global instructions: {instructions}")
    if table_instructions:
        logger.info(f"Table-specific instructions: {table_instructions}")
    
    engine = workflow.build()
    result = anyio.run(engine.run, initial_context)
    logger.info("Database analysis and sample generation completed successfully 🎉")
    return result

if __name__ == "__main__":
    import typer
    from typing import Optional, List
    
    def main(
        source_db_url: str = "postgresql://quantadbu:azerty1234@localhost:5432/test",
        target_db_url: Optional[str] = None,
        tables: Optional[List[str]] = None,
        model: str = "gemini/gemini-2.0-flash",
        instructions: Optional[str] = None,
        table_instructions: Optional[str] = None
    ):
        """
        Analyze database schema and generate sample data.
        
        Args:
            source_db_url: Source database URL
            target_db_url: Target database URL (if not provided, will create a new database with _sample suffix)
            tables: Optional list of specific tables to analyze
            model: LLM model to use
            instructions: Natural language instructions for data generation
                Example: "Generate realistic medical staff data with various roles and departments"
            table_instructions: JSON-formatted string with table-specific instructions
                Example: '{"users": "Make sure to include admin roles", "appointments": "Include emergency slots"}'
        """
        # Parse table instructions if provided
        table_instructions_dict = None
        if table_instructions:
            try:
                import json
                table_instructions_dict = json.loads(table_instructions)
            except json.JSONDecodeError:
                logger.warning("Invalid table instructions format. Should be a JSON string.")
        
        # If target_db_url not provided, create one with _sample suffix
        if not target_db_url:
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(source_db_url)
            path_parts = parsed.path.rsplit('/', 1)
            new_db_name = f"{path_parts[-1]}_sample"
            target_db_url = source_db_url.replace(path_parts[-1], new_db_name)
            
            # Create the target database if it doesn't exist
            engine = create_engine(source_db_url.replace(path_parts[-1], 'postgres'))
            with engine.connect() as conn:
                conn.execute(text("commit"))
                try:
                    conn.execute(text(f"CREATE DATABASE {new_db_name}"))
                    logger.info(f"Created target database: {new_db_name}")
                except Exception as e:
                    if "already exists" not in str(e):
                        raise
                    logger.info(f"Target database {new_db_name} already exists")
        
        print("🚀 Starting database analysis and sample data generation...")
        print(f"Source DB: {source_db_url}")
        print(f"Target DB: {target_db_url}")
        if tables:
            print(f"Analyzing tables: {', '.join(tables)}")
        else:
            print("Analyzing all tables")
        if instructions:
            print(f"\n📝 Instructions: {instructions}")
        if table_instructions_dict:
            print("\n📋 Table-specific instructions:")
            for table, instr in table_instructions_dict.items():
                print(f"- {table}: {instr}")
            
        try:
            result = analyze_and_generate_samples(
                source_db_url=source_db_url,
                target_db_url=target_db_url,
                model=model,
                specific_tables=tables,
                instructions=instructions,
                table_instructions=table_instructions_dict
            )
            
            print("\n✅ Process completed successfully!")
            
            # Show some statistics from target database
            engine = create_engine(target_db_url)
            with engine.connect() as conn:
                print("\n📊 Generated Data Statistics:")
                table_list = tables if tables else [
                    row[0] for row in conn.execute(text(
                        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
                    ))
                ]
                
                for table in table_list:
                    count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                    print(f"- {table}: {count} records")
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            raise

    typer.run(main)
