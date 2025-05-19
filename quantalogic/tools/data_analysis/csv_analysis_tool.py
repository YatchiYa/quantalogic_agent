"""CSV Analysis Tool for analyzing and visualizing data from CSV files."""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import uuid

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from quantalogic.tools.tool import Tool, ToolArgument
from quantalogic.tools.llm_tool import LLMTool
import webbrowser
import datetime

class CSVAnalysisTool(Tool):
    """Advanced CSV analysis and visualization tool with LLM-powered insights."""

    class Config(BaseModel):
        """Configuration for the CSV Analysis Tool."""
        
        csv_paths: List[str] = Field(
            ..., 
            description="List of paths to CSV files to analyze"
        )
        output_dir: str = Field(
            default="./analysis_output",
            description="Directory to save analysis results and visualizations"
        )
        analysis_type: str = Field(
            default="comprehensive",
            description="Type of analysis to perform (basic, comprehensive, custom)"
        )
        custom_columns: Optional[List[str]] = Field(
            default=None,
            description="List of specific columns to analyze (only used if analysis_type is 'custom')"
        )
        visualization_types: List[str] = Field(
            default=["histogram", "scatter", "correlation", "boxplot", "bar", "pie", "3d_scatter", "sunburst", "treemap", "heatmap", "parallel", "violin"],
            description="Types of visualizations to generate"
        )
        use_llm: bool = Field(
            default=True,
            description="Whether to use LLM for enhanced analysis"
        )
        llm_model: str = Field(
            default="openai/gpt-4o-mini",
            description="LLM model to use for analysis"
        )
        
        @field_validator("analysis_type")
        def validate_analysis_type(cls, v: str) -> str:
            valid_types = ["basic", "comprehensive", "custom"]
            if v not in valid_types:
                raise ValueError(f"analysis_type must be one of: {', '.join(valid_types)}")
            return v
        
        @field_validator("visualization_types")
        def validate_visualization_types(cls, v: List[str]) -> List[str]:
            valid_types = [
                "histogram", "scatter", "correlation", "boxplot", "bar", "pie", "3d_scatter", "sunburst", "treemap", "heatmap", "parallel", "violin"
            ]
            for viz_type in v:
                if viz_type not in valid_types:
                    raise ValueError(f"visualization_types must be from: {', '.join(valid_types)}")
            return v

    name: str = "csv_analysis"
    description: str = (
        "Advanced CSV analysis and visualization tool with LLM-powered insights. "
        "Analyzes one or more CSV files, generates statistical insights, "
        "creates interactive visualizations saved as HTML and static images, "
        "and provides AI-powered analysis of patterns and trends. "
        "Supports various types of analysis and visualization options."
    )
    arguments: list = [
        ToolArgument(
            name="csv_paths",
            arg_type="string",
            description="Comma-separated list of paths to CSV files to analyze",
            required=True,
            example="/path/to/data1.csv,/path/to/data2.csv"
        ),
        ToolArgument(
            name="output_dir",
            arg_type="string",
            description="Directory to save analysis results and visualizations",
            required=False,
            default="./analysis_output"
        ),
        ToolArgument(
            name="analysis_type",
            arg_type="string",
            description="Type of analysis to perform (basic, comprehensive, custom)",
            required=False,
            default="comprehensive"
        ),
        ToolArgument(
            name="custom_columns",
            arg_type="string",
            description="Comma-separated list of specific columns to analyze (only used if analysis_type is 'custom')",
            required=False,
            default=""
        ),
        ToolArgument(
            name="visualization_types",
            arg_type="string",
            description="Comma-separated list of visualization types to generate (histogram, scatter, correlation, boxplot, bar, pie, 3d_scatter, sunburst, treemap, heatmap, parallel, violin)",
            required=False,
            default="histogram,scatter,correlation,boxplot,bar,pie,3d_scatter,sunburst,treemap,heatmap,parallel,violin"
        ),
        ToolArgument(
            name="use_llm",
            arg_type="boolean",
            description="Whether to use LLM for enhanced analysis",
            required=False,
            default="true"
        ),
        ToolArgument(
            name="llm_model",
            arg_type="string",
            description="LLM model to use for analysis",
            required=False,
            default="openai/gpt-4o-mini"
        )
    ]

    def __init__(self, **kwargs):
        """Initialize the CSVAnalysisTool."""
        super().__init__(**kwargs)
        # Set default style for visualizations
        sns.set_theme(style="whitegrid")
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.rcParams["figure.dpi"] = 100
        
        # Set Plotly default template
        pio.templates.default = "plotly_white"
        
        # Initialize LLM tool
        self.llm = None  # Will be initialized when needed

    def _load_csv_files(self, csv_paths: List[str]) -> Dict[str, pd.DataFrame]:
        """Load CSV files into pandas DataFrames.
        
        Args:
            csv_paths: List of paths to CSV files
            
        Returns:
            Dictionary mapping filenames to DataFrames
        """
        dataframes = {}
        
        for path in csv_paths:
            try:
                # Validate path exists
                if not os.path.exists(path):
                    logger.error(f"CSV file not found: {path}")
                    continue
                
                # Load CSV file
                filename = os.path.basename(path)
                df = pd.read_csv(path)
                
                # Basic validation
                if df.empty:
                    logger.warning(f"CSV file is empty: {path}")
                    continue
                
                dataframes[filename] = df
                logger.info(f"Successfully loaded {filename} with {len(df)} rows and {len(df.columns)} columns")
                
            except Exception as e:
                logger.error(f"Error loading CSV file {path}: {e}")
        
        return dataframes

    def _create_output_directory(self, output_dir: str) -> str:
        """Create output directory for analysis results.
        
        Args:
            output_dir: Base directory for output
            
        Returns:
            Path to the created directory
        """
        # Create a unique subdirectory for this analysis run
        run_id = uuid.uuid4().hex[:8]
        full_path = os.path.join(output_dir, f"analysis_{run_id}")
        
        try:
            os.makedirs(full_path, exist_ok=True)
            logger.info(f"Created output directory: {full_path}")
            return full_path
        except Exception as e:
            logger.error(f"Error creating output directory: {e}")
            # Fall back to the base directory
            os.makedirs(output_dir, exist_ok=True)
            return output_dir

    def _get_basic_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic statistics for a DataFrame.
        
        Args:
            df: Pandas DataFrame to analyze
            
        Returns:
            Dictionary of basic statistics
        """
        stats = {}
        
        # Basic dataset info
        stats["row_count"] = len(df)
        stats["column_count"] = len(df.columns)
        stats["column_names"] = df.columns.tolist()
        
        # Missing values
        missing_values = df.isnull().sum().to_dict()
        stats["missing_values"] = {k: int(v) for k, v in missing_values.items() if v > 0}
        stats["missing_percentage"] = {k: f"{(v/len(df)*100):.2f}%" for k, v in missing_values.items() if v > 0}
        
        # Data types
        stats["data_types"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        # Numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        stats["numeric_columns"] = numeric_cols
        
        # Categorical columns
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        stats["categorical_columns"] = categorical_cols
        
        # Date columns (basic detection)
        date_cols = []
        for col in df.columns:
            if "date" in col.lower() or "time" in col.lower():
                try:
                    pd.to_datetime(df[col], errors="raise")
                    date_cols.append(col)
                except:
                    pass
        stats["date_columns"] = date_cols
        
        return stats

    def _get_numeric_stats(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Calculate statistics for numeric columns.
        
        Args:
            df: Pandas DataFrame to analyze
            columns: Optional list of specific columns to analyze
            
        Returns:
            Dictionary of numeric column statistics
        """
        numeric_stats = {}
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter columns if specified
        if columns:
            numeric_cols = [col for col in numeric_cols if col in columns]
        
        # Calculate statistics for each numeric column
        for col in numeric_cols:
            col_stats = {}
            
            # Basic statistics
            col_stats["min"] = float(df[col].min())
            col_stats["max"] = float(df[col].max())
            col_stats["mean"] = float(df[col].mean())
            col_stats["median"] = float(df[col].median())
            col_stats["std"] = float(df[col].std())
            
            # Percentiles
            percentiles = [0.1, 0.25, 0.5, 0.75, 0.9]
            for p in percentiles:
                col_stats[f"percentile_{int(p*100)}"] = float(df[col].quantile(p))
            
            # Count of zeros and negative values
            col_stats["zero_count"] = int((df[col] == 0).sum())
            col_stats["negative_count"] = int((df[col] < 0).sum())
            
            # Add to overall stats
            numeric_stats[col] = col_stats
        
        return numeric_stats

    def _get_categorical_stats(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Calculate statistics for categorical columns.
        
        Args:
            df: Pandas DataFrame to analyze
            columns: Optional list of specific columns to analyze
            
        Returns:
            Dictionary of categorical column statistics
        """
        categorical_stats = {}
        
        # Get categorical columns
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        
        # Filter columns if specified
        if columns:
            categorical_cols = [col for col in categorical_cols if col in columns]
        
        # Calculate statistics for each categorical column
        for col in categorical_cols:
            col_stats = {}
            
            # Value counts (top 10)
            value_counts = df[col].value_counts().head(10).to_dict()
            col_stats["value_counts"] = {str(k): int(v) for k, v in value_counts.items()}
            
            # Unique values count
            col_stats["unique_count"] = int(df[col].nunique())
            
            # Most common and least common values
            if col_stats["unique_count"] > 0:
                most_common = df[col].value_counts().index[0]
                col_stats["most_common"] = str(most_common)
                
                if col_stats["unique_count"] > 1:
                    least_common = df[col].value_counts().index[-1]
                    col_stats["least_common"] = str(least_common)
            
            # Add to overall stats
            categorical_stats[col] = col_stats
        
        return categorical_stats

    def _generate_histogram(self, df: pd.DataFrame, column: str, output_dir: str) -> str:
        """Generate a histogram for a numeric column.
        
        Args:
            df: Pandas DataFrame
            column: Column name to visualize
            output_dir: Directory to save the visualization
            
        Returns:
            Path to the saved image file
        """
        plt.figure()
        
        # Create histogram with KDE
        sns.histplot(df[column], kde=True)
        
        # Add title and labels
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        
        # Save figure
        filename = f"histogram_{column.replace(' ', '_')}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        plt.close()
        
        return filepath

    def _generate_boxplot(self, df: pd.DataFrame, column: str, output_dir: str) -> str:
        """Generate a boxplot for a numeric column.
        
        Args:
            df: Pandas DataFrame
            column: Column name to visualize
            output_dir: Directory to save the visualization
            
        Returns:
            Path to the saved image file
        """
        plt.figure()
        
        # Create boxplot
        sns.boxplot(x=df[column])
        
        # Add title and labels
        plt.title(f"Boxplot of {column}")
        plt.xlabel(column)
        
        # Save figure
        filename = f"boxplot_{column.replace(' ', '_')}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        plt.close()
        
        return filepath

    def _generate_scatter_plot(self, df: pd.DataFrame, x_col: str, y_col: str, output_dir: str) -> str:
        """Generate a scatter plot for two numeric columns.
        
        Args:
            df: Pandas DataFrame
            x_col: Column name for x-axis
            y_col: Column name for y-axis
            output_dir: Directory to save the visualization
            
        Returns:
            Path to the saved image file
        """
        plt.figure()
        
        # Create scatter plot
        sns.scatterplot(data=df, x=x_col, y=y_col)
        
        # Add title and labels
        plt.title(f"Scatter Plot: {x_col} vs {y_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        
        # Save figure
        filename = f"scatter_{x_col.replace(' ', '_')}_{y_col.replace(' ', '_')}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        plt.close()
        
        return filepath

    def _generate_correlation_heatmap(self, df: pd.DataFrame, output_dir: str) -> str:
        """Generate a correlation heatmap for numeric columns.
        
        Args:
            df: Pandas DataFrame
            output_dir: Directory to save the visualization
            
        Returns:
            Path to the saved image file
        """
        # Get numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Skip if not enough numeric columns
        if len(numeric_df.columns) < 2:
            return None
        
        plt.figure(figsize=(12, 10))
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        
        # Add title
        plt.title("Correlation Matrix")
        plt.tight_layout()
        
        # Save figure
        filename = "correlation_heatmap.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        plt.close()
        
        return filepath

    def _generate_bar_chart(self, df: pd.DataFrame, column: str, output_dir: str) -> str:
        """Generate a bar chart for a categorical column.
        
        Args:
            df: Pandas DataFrame
            column: Column name to visualize
            output_dir: Directory to save the visualization
            
        Returns:
            Path to the saved image file
        """
        # Get value counts (top 10)
        value_counts = df[column].value_counts().head(10)
        
        plt.figure(figsize=(12, 6))
        
        # Create bar chart
        sns.barplot(x=value_counts.index, y=value_counts.values)
        
        # Add title and labels
        plt.title(f"Top 10 Values in {column}")
        plt.xlabel(column)
        plt.ylabel("Count")
        
        # Rotate x-axis labels if needed
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        # Save figure
        filename = f"bar_{column.replace(' ', '_')}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        plt.close()
        
        return filepath

    def _generate_pie_chart(self, df: pd.DataFrame, column: str, output_dir: str) -> str:
        """Generate a pie chart for a categorical column.
        
        Args:
            df: Pandas DataFrame
            column: Column name to visualize
            output_dir: Directory to save the visualization
            
        Returns:
            Path to the saved image file
        """
        # Get value counts (top 5)
        value_counts = df[column].value_counts().head(5)
        
        # Add "Other" category if needed
        if df[column].nunique() > 5:
            other_count = df[column].value_counts().iloc[5:].sum()
            value_counts["Other"] = other_count
        
        plt.figure()
        
        # Create pie chart
        plt.pie(value_counts.values, labels=value_counts.index, autopct="%1.1f%%", startangle=90)
        plt.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle
        
        # Add title
        plt.title(f"Distribution of {column}")
        
        # Save figure
        filename = f"pie_{column.replace(' ', '_')}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        plt.close()
        
        return filepath

    def _generate_line_chart(self, df: pd.DataFrame, x_col: str, y_col: str, output_dir: str) -> str:
        """Generate a line chart for two columns.
        
        Args:
            df: Pandas DataFrame
            x_col: Column name for x-axis
            y_col: Column name for y-axis
            output_dir: Directory to save the visualization
            
        Returns:
            Path to the saved image file
        """
        plt.figure()
        
        # Sort by x column
        sorted_df = df.sort_values(by=x_col)
        
        # Create line chart
        sns.lineplot(data=sorted_df, x=x_col, y=y_col)
        
        # Add title and labels
        plt.title(f"Line Chart: {y_col} over {x_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        
        # Save figure
        filename = f"line_{x_col.replace(' ', '_')}_{y_col.replace(' ', '_')}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        plt.close()
        
        return filepath

    def _generate_3d_scatter_plot(self, df: pd.DataFrame, x_col: str, y_col: str, z_col: str, output_dir: str) -> str:
        """Generate a 3D scatter plot for three numeric columns.
        
        Args:
            df: Pandas DataFrame
            x_col: Column name for x-axis
            y_col: Column name for y-axis
            z_col: Column name for z-axis
            output_dir: Directory to save the visualization
            
        Returns:
            Path to the saved HTML file
        """
        # Create a more advanced 3D scatter plot with better styling
        fig = px.scatter_3d(
            df, 
            x=x_col, 
            y=y_col, 
            z=z_col,
            opacity=0.7,
            color_discrete_sequence=px.colors.qualitative.G10
        )
        
        # Improve layout
        fig.update_layout(
            title=f"3D Scatter Plot: {x_col} vs {y_col} vs {z_col}",
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col,
                aspectmode='cube'
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        # Save figure as both HTML (interactive) and PNG (static)
        html_filename = f"3d_scatter_{x_col.replace(' ', '_')}_{y_col.replace(' ', '_')}_{z_col.replace(' ', '_')}.html"
        html_filepath = os.path.join(output_dir, html_filename)
        fig.write_html(html_filepath, include_plotlyjs='cdn')
        
        png_filename = f"3d_scatter_{x_col.replace(' ', '_')}_{y_col.replace(' ', '_')}_{z_col.replace(' ', '_')}.png"
        png_filepath = os.path.join(output_dir, png_filename)
        fig.write_image(png_filepath, width=800, height=600, scale=2)
        
        return html_filepath

    def _generate_sunburst_chart(self, df: pd.DataFrame, column: str, output_dir: str) -> str:
        """Generate a sunburst chart for a categorical column.
        
        Args:
            df: Pandas DataFrame
            column: Column name to visualize
            output_dir: Directory to save the visualization
            
        Returns:
            Path to the saved HTML file
        """
        try:
            # Get value counts
            value_counts = df[column].value_counts().reset_index()
            value_counts.columns = [column, 'count']
            
            # Create a more advanced sunburst chart
            fig = px.sunburst(
                value_counts, 
                path=[column], 
                values='count',
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            
            # Improve layout
            fig.update_layout(
                title=f"Distribution of {column}",
                margin=dict(t=60, l=0, r=0, b=0)
            )
            
            # Save figure as both HTML (interactive) and PNG (static)
            html_filename = f"sunburst_{column.replace(' ', '_')}.html"
            html_filepath = os.path.join(output_dir, html_filename)
            fig.write_html(html_filepath, include_plotlyjs='cdn')
            
            png_filename = f"sunburst_{column.replace(' ', '_')}.png"
            png_filepath = os.path.join(output_dir, png_filename)
            fig.write_image(png_filename, width=800, height=800, scale=2)
            
            return html_filepath
        except Exception as e:
            logger.error(f"Error generating sunburst chart for {column}: {e}")
            return None

    def _generate_treemap_chart(self, df: pd.DataFrame, column: str, output_dir: str) -> str:
        """Generate a treemap chart for a categorical column.
        
        Args:
            df: Pandas DataFrame
            column: Column name to visualize
            output_dir: Directory to save the visualization
            
        Returns:
            Path to the saved HTML file
        """
        try:
            # Get value counts
            value_counts = df[column].value_counts().reset_index()
            value_counts.columns = [column, 'count']
            
            # Create a more advanced treemap chart
            fig = px.treemap(
                value_counts, 
                path=[column], 
                values='count',
                color='count',
                color_continuous_scale='Viridis',
                hover_data={column: True, 'count': True}
            )
            
            # Improve layout
            fig.update_layout(
                title=f"Treemap of {column}",
                margin=dict(t=50, l=0, r=0, b=0)
            )
            
            # Save figure as both HTML (interactive) and PNG (static)
            html_filename = f"treemap_{column.replace(' ', '_')}.html"
            html_filepath = os.path.join(output_dir, html_filename)
            fig.write_html(html_filepath, include_plotlyjs='cdn')
            
            png_filename = f"treemap_{column.replace(' ', '_')}.png"
            png_filepath = os.path.join(output_dir, png_filename)
            fig.write_image(png_filename, width=900, height=700, scale=2)
            
            return html_filepath
        except Exception as e:
            logger.error(f"Error generating treemap chart for {column}: {e}")
            return None
        
    def _generate_heatmap(self, df: pd.DataFrame, output_dir: str) -> str:
        """Generate an enhanced correlation heatmap for numeric columns.
        
        Args:
            df: Pandas DataFrame
            output_dir: Directory to save the visualization
            
        Returns:
            Path to the saved HTML file
        """
        # Get numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Skip if not enough numeric columns
        if len(numeric_df.columns) < 2:
            return None
            
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Create an interactive heatmap with Plotly
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect="auto",
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1
        )
        
        # Improve layout
        fig.update_layout(
            title="Correlation Matrix Heatmap",
            xaxis_title="Features",
            yaxis_title="Features",
            height=800,
            width=900
        )
        
        # Save figure as both HTML (interactive) and PNG (static)
        html_filename = "correlation_heatmap_interactive.html"
        html_filepath = os.path.join(output_dir, html_filename)
        fig.write_html(html_filepath, include_plotlyjs='cdn')
        
        png_filename = "correlation_heatmap.png"
        png_filepath = os.path.join(output_dir, png_filename)
        fig.write_image(png_filename, width=900, height=800, scale=2)
        
        return html_filepath
        
    def _generate_parallel_coordinates(self, df: pd.DataFrame, output_dir: str) -> str:
        """Generate a parallel coordinates plot for numeric columns.
        
        Args:
            df: Pandas DataFrame
            output_dir: Directory to save the visualization
            
        Returns:
            Path to the saved HTML file
        """
        # Get numeric columns (limit to 10 to avoid overcrowding)
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 10:
            # Select columns with highest variance
            variances = numeric_df.var().sort_values(ascending=False)
            selected_columns = variances.index[:10].tolist()
            numeric_df = numeric_df[selected_columns]
            
        # Skip if not enough numeric columns
        if len(numeric_df.columns) < 3:
            return None
            
        # Create a parallel coordinates plot
        fig = px.parallel_coordinates(
            numeric_df,
            color=numeric_df.columns[0],
            color_continuous_scale=px.colors.diverging.Tealrose,
            labels={col: col for col in numeric_df.columns}
        )
        
        # Improve layout
        fig.update_layout(
            title="Parallel Coordinates Plot",
            margin=dict(l=80, r=80, t=80, b=80)
        )
        
        # Save figure as both HTML (interactive) and PNG (static)
        html_filename = "parallel_coordinates.html"
        html_filepath = os.path.join(output_dir, html_filename)
        fig.write_html(html_filepath, include_plotlyjs='cdn')
        
        png_filename = "parallel_coordinates.png"
        png_filepath = os.path.join(output_dir, png_filename)
        fig.write_image(png_filename, width=1200, height=600, scale=2)
        
        return html_filepath
        
    def _generate_violin_plot(self, df: pd.DataFrame, column: str, output_dir: str) -> str:
        """Generate a violin plot for a numeric column.
        
        Args:
            df: Pandas DataFrame
            column: Column name to visualize
            output_dir: Directory to save the visualization
            
        Returns:
            Path to the saved HTML file
        """
        # Create a violin plot
        fig = px.violin(
            df, 
            y=column,
            box=True,
            points="all",
            hover_data=df.columns[:5].tolist()
        )
        
        # Improve layout
        fig.update_layout(
            title=f"Distribution of {column}",
            yaxis_title=column,
            height=600,
            width=800
        )
        
        # Save figure as both HTML (interactive) and PNG (static)
        html_filename = f"violin_{column.replace(' ', '_')}.html"
        html_filepath = os.path.join(output_dir, html_filename)
        fig.write_html(html_filepath, include_plotlyjs='cdn')
        
        png_filename = f"violin_{column.replace(' ', '_')}.png"
        png_filepath = os.path.join(output_dir, png_filename)
        fig.write_image(png_filename, width=800, height=600, scale=2)
        
        return html_filepath

    def _generate_visualizations(
        self, 
        df: pd.DataFrame, 
        output_dir: str, 
        viz_types: List[str],
        columns: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """Generate visualizations based on the specified types.
        
        Args:
            df: Pandas DataFrame
            output_dir: Directory to save visualizations
            viz_types: List of visualization types to generate
            columns: Optional list of specific columns to visualize
            
        Returns:
            Dictionary mapping visualization types to lists of image paths
        """
        visualization_paths = {viz_type: [] for viz_type in viz_types}
        
        # Get numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        
        # Filter columns if specified
        if columns:
            numeric_cols = [col for col in numeric_cols if col in columns]
            categorical_cols = [col for col in categorical_cols if col in columns]
        
        # Generate visualizations based on types
        for viz_type in viz_types:
            if viz_type == "histogram":
                # Generate histograms for numeric columns
                for col in numeric_cols:
                    try:
                        filepath = self._generate_histogram(df, col, output_dir)
                        if filepath:
                            visualization_paths["histogram"].append(filepath)
                    except Exception as e:
                        logger.error(f"Error generating histogram for {col}: {e}")
            
            elif viz_type == "boxplot":
                # Generate boxplots for numeric columns
                for col in numeric_cols:
                    try:
                        filepath = self._generate_boxplot(df, col, output_dir)
                        if filepath:
                            visualization_paths["boxplot"].append(filepath)
                    except Exception as e:
                        logger.error(f"Error generating boxplot for {col}: {e}")
            
            elif viz_type == "scatter":
                # Generate scatter plots for pairs of numeric columns (limit to 10 pairs)
                if len(numeric_cols) >= 2:
                    pairs = []
                    for i in range(min(len(numeric_cols), 5)):
                        for j in range(i+1, min(len(numeric_cols), 5)):
                            pairs.append((numeric_cols[i], numeric_cols[j]))
                    
                    for x_col, y_col in pairs:
                        try:
                            filepath = self._generate_scatter_plot(df, x_col, y_col, output_dir)
                            if filepath:
                                visualization_paths["scatter"].append(filepath)
                        except Exception as e:
                            logger.error(f"Error generating scatter plot for {x_col} vs {y_col}: {e}")
            
            elif viz_type == "correlation":
                # Generate correlation heatmap
                try:
                    filepath = self._generate_correlation_heatmap(df, output_dir)
                    if filepath:
                        visualization_paths["correlation"].append(filepath)
                except Exception as e:
                    logger.error(f"Error generating correlation heatmap: {e}")
            
            elif viz_type == "bar":
                # Generate bar charts for categorical columns
                for col in categorical_cols:
                    try:
                        filepath = self._generate_bar_chart(df, col, output_dir)
                        if filepath:
                            visualization_paths["bar"].append(filepath)
                    except Exception as e:
                        logger.error(f"Error generating bar chart for {col}: {e}")
            
            elif viz_type == "pie":
                # Generate pie charts for categorical columns
                for col in categorical_cols:
                    try:
                        filepath = self._generate_pie_chart(df, col, output_dir)
                        if filepath:
                            visualization_paths["pie"].append(filepath)
                    except Exception as e:
                        logger.error(f"Error generating pie chart for {col}: {e}")
            
            elif viz_type == "line":
                # Generate line charts for numeric columns with potential date columns as x-axis
                date_cols = []
                for col in df.columns:
                    if "date" in col.lower() or "time" in col.lower():
                        try:
                            df[col] = pd.to_datetime(df[col])
                            date_cols.append(col)
                        except:
                            pass
                
                # If no date columns, use the first numeric column as x-axis
                if not date_cols and numeric_cols:
                    x_col = numeric_cols[0]
                    for y_col in numeric_cols[1:]:
                        try:
                            filepath = self._generate_line_chart(df, x_col, y_col, output_dir)
                            if filepath:
                                visualization_paths["line"].append(filepath)
                        except Exception as e:
                            logger.error(f"Error generating line chart for {x_col} vs {y_col}: {e}")
                else:
                    # Use date columns as x-axis
                    for x_col in date_cols:
                        for y_col in numeric_cols:
                            try:
                                filepath = self._generate_line_chart(df, x_col, y_col, output_dir)
                                if filepath:
                                    visualization_paths["line"].append(filepath)
                            except Exception as e:
                                logger.error(f"Error generating line chart for {x_col} vs {y_col}: {e}")
            
            elif viz_type == "3d_scatter":
                # Generate 3D scatter plots for three numeric columns (limit to 10 triplets)
                if len(numeric_cols) >= 3:
                    triplets = []
                    for i in range(min(len(numeric_cols), 5)):
                        for j in range(i+1, min(len(numeric_cols), 5)):
                            for k in range(j+1, min(len(numeric_cols), 5)):
                                triplets.append((numeric_cols[i], numeric_cols[j], numeric_cols[k]))
                    
                    for x_col, y_col, z_col in triplets:
                        try:
                            filepath = self._generate_3d_scatter_plot(df, x_col, y_col, z_col, output_dir)
                            if filepath:
                                visualization_paths["3d_scatter"].append(filepath)
                        except Exception as e:
                            logger.error(f"Error generating 3D scatter plot for {x_col} vs {y_col} vs {z_col}: {e}")
            
            elif viz_type == "sunburst":
                # Generate sunburst charts for categorical columns
                for col in categorical_cols:
                    try:
                        filepath = self._generate_sunburst_chart(df, col, output_dir)
                        if filepath:
                            visualization_paths["sunburst"].append(filepath)
                    except Exception as e:
                        logger.error(f"Error generating sunburst chart for {col}: {e}")
            
            elif viz_type == "treemap":
                # Generate treemap charts for categorical columns
                for col in categorical_cols:
                    try:
                        filepath = self._generate_treemap_chart(df, col, output_dir)
                        if filepath:
                            visualization_paths["treemap"].append(filepath)
                    except Exception as e:
                        logger.error(f"Error generating treemap chart for {col}: {e}")
            
            elif viz_type == "heatmap":
                # Generate heatmap
                try:
                    filepath = self._generate_heatmap(df, output_dir)
                    if filepath:
                        visualization_paths["heatmap"].append(filepath)
                except Exception as e:
                    logger.error(f"Error generating heatmap: {e}")
            
            elif viz_type == "parallel":
                # Generate parallel coordinates plot
                try:
                    filepath = self._generate_parallel_coordinates(df, output_dir)
                    if filepath:
                        visualization_paths["parallel"].append(filepath)
                except Exception as e:
                    logger.error(f"Error generating parallel coordinates plot: {e}")
            
            elif viz_type == "violin":
                # Generate violin plots for numeric columns
                for col in numeric_cols:
                    try:
                        filepath = self._generate_violin_plot(df, col, output_dir)
                        if filepath:
                            visualization_paths["violin"].append(filepath)
                    except Exception as e:
                        logger.error(f"Error generating violin plot for {col}: {e}")
        
        return visualization_paths

    def _analyze_dataframe(
        self, 
        df: pd.DataFrame, 
        filename: str,
        output_dir: str,
        analysis_type: str,
        custom_columns: Optional[List[str]] = None,
        visualization_types: List[str] = ["histogram", "scatter", "correlation", "boxplot"],
        use_llm: bool = False,
        llm_model: str = "openai/gpt-4o-mini"
    ) -> Dict[str, Any]:
        """Analyze a single DataFrame and generate visualizations.
        
        Args:
            df: Pandas DataFrame to analyze
            filename: Name of the CSV file
            output_dir: Directory to save visualizations
            analysis_type: Type of analysis to perform
            custom_columns: Optional list of specific columns to analyze
            visualization_types: List of visualization types to generate
            use_llm: Whether to use LLM for enhanced analysis
            llm_model: LLM model to use for analysis
            
        Returns:
            Dictionary of analysis results
        """
        result = {
            "filename": filename,
            "analysis_type": analysis_type,
            "basic_stats": self._get_basic_stats(df)
        }
        
        # Filter columns if custom analysis
        columns_to_analyze = None
        if analysis_type == "custom" and custom_columns:
            columns_to_analyze = [col for col in custom_columns if col in df.columns]
            result["analyzed_columns"] = columns_to_analyze
        
        # Add numeric statistics
        if analysis_type in ["comprehensive", "custom"]:
            result["numeric_stats"] = self._get_numeric_stats(df, columns_to_analyze)
            result["categorical_stats"] = self._get_categorical_stats(df, columns_to_analyze)
        
        # Generate visualizations
        visualization_paths = self._generate_visualizations(
            df, output_dir, visualization_types, columns_to_analyze
        )
        
        # Add visualization paths to result
        result["visualizations"] = {}
        for viz_type, paths in visualization_paths.items():
            if paths:
                result["visualizations"][viz_type] = paths
        
        # Add LLM analysis if requested
        if use_llm:
            try:
                llm_analysis = self._get_llm_analysis(df, filename, analysis_type, columns_to_analyze)
                result["llm_analysis"] = llm_analysis
            except Exception as e:
                logger.error(f"Error in LLM analysis: {e}")
                result["llm_analysis_error"] = str(e)
        
        return result

    def _get_llm_analysis(
        self, 
        df: pd.DataFrame, 
        filename: str, 
        analysis_type: str,
        columns_to_analyze: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get LLM-powered analysis of the DataFrame.
        
        Args:
            df: Pandas DataFrame to analyze
            filename: Name of the CSV file
            analysis_type: Type of analysis performed
            columns_to_analyze: Optional list of specific columns analyzed
            
        Returns:
            Dictionary containing LLM analysis results
        """
        # Initialize LLM tool if not already initialized
        if self.llm is None:
            self.llm = LLMTool(model_name=self.config.llm_model)
        
        # Prepare data summary for LLM
        data_summary = self._prepare_data_summary_for_llm(df, filename, columns_to_analyze)
        
        # Create system prompt
        system_prompt = (
            "You are an expert data analyst specializing in CSV data analysis. "
            "Your task is to analyze the provided dataset summary and provide detailed insights. "
            "Focus on identifying patterns, correlations, anomalies, and potential insights. "
            "Be specific, detailed, and provide actionable recommendations based on the data."
        )
        
        # Create user prompt
        prompt = f"""
# Dataset Analysis Request

I need a comprehensive analysis of the following dataset:

## Dataset Information
- Filename: {filename}
- Rows: {len(df)}
- Columns: {len(df.columns)}

## Data Summary
{data_summary}

Please provide the following analysis:

1. **Key Insights**: What are the most important patterns or findings in this data?
2. **Correlations**: Identify any significant correlations or relationships between variables.
3. **Anomalies**: Are there any outliers or unusual patterns that require attention?
4. **Data Quality Issues**: Identify any potential data quality problems (missing values, inconsistencies, etc.).
5. **Recommendations**: What actions or further analyses would you recommend based on this data?
6. **Visualization Suggestions**: What additional visualizations might reveal important insights?

Please structure your response with clear sections and be specific about your findings.
"""
        
        # Get LLM response
        try:
            response = asyncio.run(self.llm.async_execute(
                system_prompt=system_prompt,
                prompt=prompt,
                temperature="0.2"  # Lower temperature for more factual analysis
            ))
            
            # Parse response into sections
            sections = self._parse_llm_response_into_sections(response)
            
            return {
                "full_analysis": response,
                "sections": sections
            }
            
        except Exception as e:
            logger.error(f"Error getting LLM analysis: {e}")
            raise ValueError(f"Error getting LLM analysis: {e}")

    def _prepare_data_summary_for_llm(
        self, 
        df: pd.DataFrame, 
        filename: str,
        columns_to_analyze: Optional[List[str]] = None
    ) -> str:
        """Prepare a summary of the DataFrame for LLM analysis.
        
        Args:
            df: Pandas DataFrame to analyze
            filename: Name of the CSV file
            columns_to_analyze: Optional list of specific columns to analyze
            
        Returns:
            String containing a summary of the data
        """
        summary_parts = []
        
        # Get column names
        if columns_to_analyze:
            columns = columns_to_analyze
        else:
            columns = df.columns.tolist()
        
        summary_parts.append(f"### Columns ({len(columns)})\n{', '.join(columns)}\n")
        
        # Get data types
        summary_parts.append("### Data Types")
        for col in columns:
            dtype = str(df[col].dtype)
            summary_parts.append(f"- {col}: {dtype}")
        summary_parts.append("")
        
        # Get basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if columns_to_analyze:
            numeric_cols = [col for col in numeric_cols if col in columns_to_analyze]
        
        if numeric_cols:
            summary_parts.append("### Numeric Columns Statistics")
            for col in numeric_cols[:10]:  # Limit to 10 columns to avoid token limits
                stats = df[col].describe()
                summary_parts.append(f"#### {col}")
                summary_parts.append(f"- Min: {stats['min']:.2f}")
                summary_parts.append(f"- Max: {stats['max']:.2f}")
                summary_parts.append(f"- Mean: {stats['mean']:.2f}")
                summary_parts.append(f"- Median: {stats['50%']:.2f}")
                summary_parts.append(f"- Std Dev: {stats['std']:.2f}")
                summary_parts.append("")
        
        # Get value counts for categorical columns
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if columns_to_analyze:
            categorical_cols = [col for col in categorical_cols if col in columns_to_analyze]
        
        if categorical_cols:
            summary_parts.append("### Categorical Columns")
            for col in categorical_cols[:5]:  # Limit to 5 columns to avoid token limits
                unique_count = df[col].nunique()
                summary_parts.append(f"#### {col} (Unique Values: {unique_count})")
                
                # Show top 5 values
                if unique_count > 0:
                    value_counts = df[col].value_counts().head(5)
                    for value, count in value_counts.items():
                        summary_parts.append(f"- {value}: {count} ({count/len(df)*100:.1f}%)")
                summary_parts.append("")
        
        # Get missing values information
        missing_values = df.isnull().sum()
        missing_cols = [col for col in columns if missing_values[col] > 0]
        
        if missing_cols:
            summary_parts.append("### Missing Values")
            for col in missing_cols:
                count = missing_values[col]
                percentage = count / len(df) * 100
                summary_parts.append(f"- {col}: {count} ({percentage:.1f}%)")
            summary_parts.append("")
        
        # Get correlation information for numeric columns
        if len(numeric_cols) >= 2:
            summary_parts.append("### Top Correlations")
            corr_matrix = df[numeric_cols].corr()
            
            # Get top 10 correlations (excluding self-correlations)
            correlations = []
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    col1 = numeric_cols[i]
                    col2 = numeric_cols[j]
                    corr_value = corr_matrix.loc[col1, col2]
                    if not np.isnan(corr_value):
                        correlations.append((col1, col2, corr_value))
            
            # Sort by absolute correlation value
            correlations.sort(key=lambda x: abs(x[2]), reverse=True)
            
            # Show top 10 correlations
            for col1, col2, corr_value in correlations[:10]:
                summary_parts.append(f"- {col1} & {col2}: {corr_value:.3f}")
            summary_parts.append("")
        
        return "\n".join(summary_parts)

    def _parse_llm_response_into_sections(self, response: str) -> Dict[str, str]:
        """Parse the LLM response into sections.
        
        Args:
            response: LLM response text
            
        Returns:
            Dictionary mapping section names to section content
        """
        sections = {}
        current_section = "introduction"
        current_content = []
        
        for line in response.split("\n"):
            # Check if line is a section header
            if line.startswith("##") or line.startswith("# "):
                # Save previous section
                if current_content:
                    sections[current_section] = "\n".join(current_content).strip()
                    current_content = []
                
                # Extract new section name
                section_name = line.strip("#").strip().lower()
                section_name = section_name.replace(" ", "_")
                current_section = section_name
            else:
                current_content.append(line)
        
        # Save the last section
        if current_content:
            sections[current_section] = "\n".join(current_content).strip()
        
        return sections

    def execute(
        self,
        csv_paths: str,
        output_dir: str = "./analysis_output",
        analysis_type: str = "comprehensive",
        custom_columns: str = "",
        visualization_types: str = "histogram,scatter,correlation,boxplot,bar,pie,3d_scatter,sunburst,treemap,heatmap,parallel,violin",
        use_llm: str = "true",
        llm_model: str = "openai/gpt-4o-mini"
    ) -> str:
        """Execute the CSV analysis tool.
        
        Args:
            csv_paths: Comma-separated list of paths to CSV files
            output_dir: Directory to save analysis results and visualizations
            analysis_type: Type of analysis to perform
            custom_columns: Comma-separated list of specific columns to analyze
            visualization_types: Comma-separated list of visualization types to generate
            use_llm: Whether to use LLM for enhanced analysis
            llm_model: LLM model to use for analysis
            
        Returns:
            JSON string with analysis results and visualization paths
        """
        try:
            # Parse input parameters
            csv_path_list = [path.strip() for path in csv_paths.split(",")]
            viz_types_list = [viz.strip() for viz in visualization_types.split(",")]
            custom_columns_list = [col.strip() for col in custom_columns.split(",")] if custom_columns else None
            should_use_llm = use_llm.lower() == "true"
            
            # Validate and convert parameters
            self.config = self.Config(
                csv_paths=csv_path_list,
                output_dir=output_dir,
                analysis_type=analysis_type,
                custom_columns=custom_columns_list,
                visualization_types=viz_types_list,
                use_llm=should_use_llm,
                llm_model=llm_model
            )
            
            # Create output directory
            output_dir = self._create_output_directory(self.config.output_dir)
            
            # Load CSV files
            dataframes = self._load_csv_files(self.config.csv_paths)
            
            if not dataframes:
                return json.dumps({"error": "No valid CSV files found"})
            
            # Analyze each DataFrame
            results = []
            for filename, df in dataframes.items():
                result = self._analyze_dataframe(
                    df,
                    filename,
                    output_dir,
                    self.config.analysis_type,
                    self.config.custom_columns,
                    self.config.visualization_types,
                    self.config.use_llm,
                    self.config.llm_model
                )
                results.append(result)
            
            # Create summary
            summary = {
                "analysis_timestamp": datetime.datetime.now().isoformat(),
                "files_analyzed": len(results),
                "output_directory": output_dir,
                "analysis_type": self.config.analysis_type,
                "visualization_types": self.config.visualization_types,
                "llm_analysis": self.config.use_llm,
                "results": results
            }
            
            # Save summary to file
            summary_path = os.path.join(output_dir, "analysis_summary.json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            
            # Create HTML report
            report_path = self._create_html_report(summary, output_dir)
            if report_path:
                summary["report_path"] = report_path
                
                # Open the report in the default browser
                try:
                    logger.info(f"Opening HTML report: {report_path}")
                    webbrowser.open(f"file://{os.path.abspath(report_path)}")
                except Exception as e:
                    logger.error(f"Error opening HTML report: {e}")
            
            logger.info(f"Analysis complete. Results saved to {output_dir}")
            
            # Return summary as JSON string
            return json.dumps(summary, indent=2)
            
        except Exception as e:
            logger.error(f"Error in CSV analysis: {e}")
            return json.dumps({"error": str(e)})

    def _create_html_report(self, summary: Dict[str, Any], output_dir: str) -> str:
        """Create an HTML report of the analysis results.
        
        Args:
            summary: Analysis summary dictionary
            output_dir: Directory to save the report
            
        Returns:
            Path to the saved HTML report
        """
        # Create a unique filename for the report
        report_filename = f"csv_analysis_report_{uuid.uuid4().hex[:8]}.html"
        report_filepath = os.path.join(output_dir, report_filename)
        
        # Start building HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>CSV Analysis Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f8f9fa;
                }}
                h1, h2, h3, h4 {{
                    color: #2c3e50;
                }}
                h1 {{
                    text-align: center;
                    padding-bottom: 15px;
                    border-bottom: 2px solid #3498db;
                    margin-bottom: 30px;
                }}
                h2 {{
                    border-bottom: 1px solid #ddd;
                    padding-bottom: 10px;
                    margin-top: 40px;
                }}
                .section {{
                    background-color: white;
                    padding: 20px;
                    margin-bottom: 30px;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    padding: 12px 15px;
                    border-bottom: 1px solid #ddd;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                    font-weight: bold;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .viz-container {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    justify-content: center;
                }}
                .viz-item {{
                    flex: 1 1 300px;
                    max-width: 100%;
                    margin-bottom: 20px;
                    background-color: white;
                    border-radius: 8px;
                    overflow: hidden;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .viz-item h4 {{
                    padding: 10px 15px;
                    margin: 0;
                    background-color: #f2f2f2;
                }}
                .viz-item img {{
                    width: 100%;
                    height: auto;
                    display: block;
                }}
                .viz-item iframe {{
                    width: 100%;
                    height: 500px;
                    border: none;
                }}
                .viz-link {{
                    display: block;
                    text-align: center;
                    padding: 10px;
                    background-color: #3498db;
                    color: white;
                    text-decoration: none;
                    font-weight: bold;
                }}
                .viz-link:hover {{
                    background-color: #2980b9;
                }}
                pre {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    overflow-x: auto;
                }}
                .key-insight {{
                    background-color: #e8f4f8;
                    padding: 15px;
                    border-left: 5px solid #3498db;
                    margin-bottom: 15px;
                }}
                .warning {{
                    background-color: #fff3cd;
                    padding: 15px;
                    border-left: 5px solid #ffc107;
                    margin-bottom: 15px;
                }}
            </style>
        </head>
        <body>
            <h1>CSV Analysis Report</h1>
            
            <div class="section">
                <h2>Dataset Overview</h2>
        """
        
        # Add dataset information
        if "dataset_info" in summary:
            dataset_info = summary["dataset_info"]
            html_content += f"""
                <p><strong>Files Analyzed:</strong> {dataset_info.get('num_files', 'N/A')}</p>
                <p><strong>Total Rows:</strong> {dataset_info.get('total_rows', 'N/A')}</p>
                <p><strong>Total Columns:</strong> {dataset_info.get('total_columns', 'N/A')}</p>
            """
        
        # Add descriptive statistics
        if "descriptive_stats" in summary:
            html_content += """
                <h3>Descriptive Statistics</h3>
                <pre>{}</pre>
            """.format(summary["descriptive_stats"].to_html())
        
        html_content += """
            </div>
        """
        
        # Add LLM analysis if available
        if "llm_analysis" in summary and summary["llm_analysis"]:
            llm_analysis = summary["llm_analysis"]
            html_content += """
            <div class="section">
                <h2>AI-Powered Analysis</h2>
            """
            
            if "key_insights" in llm_analysis:
                html_content += """
                <h3>Key Insights</h3>
                """
                for insight in llm_analysis["key_insights"]:
                    html_content += f"""
                    <div class="key-insight">
                        <p>{insight}</p>
                    </div>
                    """
            
            if "correlations" in llm_analysis:
                html_content += """
                <h3>Correlations</h3>
                <ul>
                """
                for correlation in llm_analysis["correlations"]:
                    html_content += f"<li>{correlation}</li>"
                html_content += "</ul>"
            
            if "anomalies" in llm_analysis:
                html_content += """
                <h3>Anomalies</h3>
                <ul>
                """
                for anomaly in llm_analysis["anomalies"]:
                    html_content += f"<li>{anomaly}</li>"
                html_content += "</ul>"
            
            if "data_quality" in llm_analysis:
                html_content += """
                <h3>Data Quality Issues</h3>
                <ul>
                """
                for issue in llm_analysis["data_quality"]:
                    html_content += f"<li>{issue}</li>"
                html_content += "</ul>"
            
            if "recommendations" in llm_analysis:
                html_content += """
                <h3>Recommendations</h3>
                <ul>
                """
                for rec in llm_analysis["recommendations"]:
                    html_content += f"<li>{rec}</li>"
                html_content += "</ul>"
            
            html_content += """
            </div>
            """
        
        # Add visualizations
        html_content += """
            <div class="section">
                <h2>Visualizations</h2>
        """
        
        # Group visualizations by type
        for viz_type, paths in summary["visualization_paths"].items():
            if paths:
                html_content += f"""
                <h3>{viz_type.replace('_', ' ').title()} Visualizations</h3>
                <div class="viz-container">
                """
                
                for path in paths:
                    if path and os.path.exists(path):
                        filename = os.path.basename(path)
                        title = filename.replace('.html', '').replace('.png', '').replace('_', ' ').title()
                        
                        # For HTML visualizations, create an iframe
                        if path.endswith('.html'):
                            html_content += f"""
                            <div class="viz-item">
                                <h4>{title}</h4>
                                <iframe src="{filename}" title="{title}"></iframe>
                                <a href="{filename}" target="_blank" class="viz-link">Open in Full Screen</a>
                            </div>
                            """
                        # For image visualizations
                        elif path.endswith(('.png', '.jpg', '.jpeg')):
                            html_content += f"""
                            <div class="viz-item">
                                <h4>{title}</h4>
                                <img src="{filename}" alt="{title}">
                                <a href="{filename}" target="_blank" class="viz-link">View Full Size</a>
                            </div>
                            """
                
                html_content += """
                </div>
                """
        
        html_content += """
            </div>
            
            <div class="section">
                <h2>Conclusion</h2>
                <p>This report provides a comprehensive analysis of the CSV data. The visualizations and statistics 
                offer insights into the data patterns, distributions, and relationships. For more detailed analysis, 
                consider exploring specific features or relationships of interest.</p>
            </div>
            
            <footer style="text-align: center; margin-top: 50px; color: #777; font-size: 0.9em;">
                <p>Generated by CSV Analysis Tool | Quantalogic</p>
            </footer>
        </body>
        </html>
        """
        
        # Write HTML content to file
        with open(report_filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report created at: {report_filepath}")
        return report_filepath

    async def async_execute(
        self,
        csv_paths: str,
        output_dir: str = "./analysis_output",
        analysis_type: str = "comprehensive",
        custom_columns: str = "",
        visualization_types: str = "histogram,scatter,correlation,boxplot,bar,pie,3d_scatter,sunburst,treemap,heatmap,parallel,violin",
        use_llm: str = "true",
        llm_model: str = "openai/gpt-4o-mini"
    ) -> str:
        """Asynchronous version of execute.
        
        Runs the synchronous execute method in a separate thread.
        
        Args:
            csv_paths: Comma-separated list of paths to CSV files
            output_dir: Directory to save analysis results and visualizations
            analysis_type: Type of analysis to perform
            custom_columns: Comma-separated list of specific columns to analyze
            visualization_types: Comma-separated list of visualization types to generate
            use_llm: Whether to use LLM for enhanced analysis
            llm_model: LLM model to use for analysis
            
        Returns:
            JSON string with analysis results and visualization paths
        """
        return await asyncio.to_thread(
            self.execute,
            csv_paths=csv_paths,
            output_dir=output_dir,
            analysis_type=analysis_type,
            custom_columns=custom_columns,
            visualization_types=visualization_types,
            use_llm=use_llm,
            llm_model=llm_model
        )


if __name__ == "__main__":
    # Example usage
    async def main():
        # Initialize tool
        tool = CSVAnalysisTool()
        
        try:
            # Example: Analyze a CSV file
            result = await tool.async_execute(
                csv_paths="/home/yarab/Téléchargements/liste-podcasts-groupe-bpce.csv",
                output_dir="./analysis_results",
                analysis_type="comprehensive",
                visualization_types="histogram,scatter,correlation,boxplot,bar,pie,3d_scatter,sunburst,treemap,heatmap,parallel,violin",
                use_llm="true"
            )
            print(result)
        except Exception as e:
            logger.error(f"Error in example: {e}")
            print(f"Error: {e}")
    
    asyncio.run(main())
