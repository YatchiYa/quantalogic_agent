#!/usr/bin/env python3
"""
JORADP Scraper - Extract and visualize legal documents from the Algerian Official Journal.

This module handles the scraping, parsing, and visualization of legal documents
from the Algerian Official Journal website (joradp.dz).
"""

import argparse
import json
import os
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import requests
from bs4 import BeautifulSoup
from loguru import logger

# Configure logger
logger.remove()
logger.add(
    "joradp_scraper.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level="INFO",
)
logger.add(lambda msg: print(msg), level="INFO")


@dataclass
class LegalDocument:
    """Represents a legal document from the Algerian Official Journal."""

    doc_id: str
    doc_type: str
    doc_number: Optional[str] = None
    date: Optional[str] = None
    ministry: Optional[str] = None
    journal_number: Optional[str] = None
    journal_date: Optional[str] = None
    journal_page: Optional[str] = None
    title: Optional[str] = None
    level: int = 0  # Hierarchical level in the document tree
    parent_id: Optional[str] = None  # Parent document ID
    modified_by: List[str] = field(default_factory=list)
    abrogated_by: List[str] = field(default_factory=list)
    modifies: List[str] = field(default_factory=list)
    abrogates: List[str] = field(default_factory=list)
    implements: List[str] = field(default_factory=list)
    implemented_by: List[str] = field(default_factory=list)
    has_rectification: List[str] = field(default_factory=list)
    is_rectification_of: List[str] = field(default_factory=list)
    # JoOpen parameters
    jo_year: Optional[str] = None
    jo_number: Optional[str] = None
    jo_page: Optional[str] = None
    jo_lang: Optional[str] = None
    bgcolor: Optional[str] = None  # Background color in the HTML, indicates level

    def to_dict(self) -> Dict:
        """Convert the document to a dictionary."""
        return asdict(self)


class JoradpScraper:
    """Scraper for the Algerian Official Journal website."""

    BASE_URL = "https://www.joradp.dz/SCRIPTS"
    SEARCH_URL = f"{BASE_URL}/Jof_Rec.dll/RecPost"
    PAGINATION_URL = f"{BASE_URL}/Jof_Rec.dll/AffPost"

    def __init__(self):
        """Initialize the scraper."""
        self.session = requests.Session()
        self.headers = {
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "accept-language": "fr-FR,fr;q=0.9",
            "cache-control": "max-age=0",
            "content-type": "application/x-www-form-urlencoded",
            "origin": "https://www.joradp.dz",
            "referer": "https://www.joradp.dz/SCRIPTS/Jof_Div.dll/RecGet",
            "sec-ch-ua": '"Chromium";v="134", "Not:A-Brand";v="24", "Google Chrome";v="134"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Linux"',
            "sec-fetch-dest": "frame",
            "sec-fetch-mode": "navigate",
            "sec-fetch-site": "same-origin",
            "sec-fetch-user": "?1",
            "upgrade-insecure-requests": "1",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
        }
        self.documents: Dict[str, LegalDocument] = {}
        self.client_id = None

    def search(
        self,
        ministry: Optional[str] = None,
        start: int = 0,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
        doc_type: int = 2,
    ) -> str:
        """
        Perform a search on the JORADP website.

        Args:
            ministry: Ministry/department to filter by
            start: Starting index for pagination
            min_date: Minimum date for search
            max_date: Maximum date for search
            doc_type: Type of document (2 = all)

        Returns:
            HTML content of the search results
        """
        data = {
            "Client": self.client_id or "112757",
            "Start": start,
            "ztyp": doc_type,
        }

        if ministry:
            data["zsec"] = ministry

        if min_date:
            data["zmin"] = min_date

        if max_date:
            data["zmax"] = max_date

        logger.info(f"Searching with parameters: {data}")
        response = self.session.post(
            self.SEARCH_URL, headers=self.headers, data=data
        )
        response.raise_for_status()

        # Extract client ID for pagination
        if not self.client_id:
            match = re.search(r'name="Client"\s+type="hidden"\s+value="(\d+)"', response.text)
            if match:
                self.client_id = match.group(1)
                logger.info(f"Extracted client ID: {self.client_id}")

        return response.text

    def paginate(self, page_number: int) -> str:
        """
        Navigate to a specific page of results.

        Args:
            page_number: Page number to navigate to

        Returns:
            HTML content of the page
        """
        if not self.client_id:
            raise ValueError("No client ID available. Run a search first.")

        # Calculate the starting index for the requested page
        # Each page shows 20 items
        start_index = (page_number - 1) * 20 + 1

        data = {
            "Client": self.client_id,
            "dfon": 3,  # Function code for pagination
            "dval": start_index,
        }

        logger.info(f"Navigating to page {page_number} (index {start_index})")
        response = self.session.post(
            self.PAGINATION_URL, headers=self.headers, data=data
        )
        response.raise_for_status()
        return response.text

    def parse_html(self, html_content: str) -> List[LegalDocument]:
        """
        Parse HTML content to extract legal documents.

        Args:
            html_content: HTML content from the JORADP website

        Returns:
            List of extracted LegalDocument objects
        """
        soup = BeautifulSoup(html_content, "html.parser")
        documents = []

        # Find the main table containing document data
        main_table = soup.find("table", {"align": "top", "width": "100%"})
        if not main_table:
            logger.warning("No main table found in HTML content")
            return documents

        # Extract total count of documents
        count_text = soup.find("td", {"id": "tex"})
        if count_text:
            match = re.search(r"Texte \d+ &agrave;(\d+) de (\d+)", count_text.text)
            if match:
                total_docs = int(match.group(2))
                logger.info(f"Total documents: {total_docs}")

        # First pass: identify all document headers and their IDs
        document_rows = []
        for row in main_table.find_all("tr"):
            bgcolor = row.get("bgcolor")
            if bgcolor:
                doc_id_cell = row.find("a")
                if doc_id_cell and "title" in doc_id_cell.attrs:
                    doc_id = doc_id_cell["title"].replace("Texte N°", "").strip()
                    level = 0
                    if bgcolor == "#78a7b9":
                        level = 0
                    elif bgcolor == "#9ec7d7":
                        level = 1
                    elif bgcolor == "#c8e7f3":
                        level = 2
                    
                    document_rows.append({
                        "row": row,
                        "doc_id": doc_id,
                        "level": level,
                        "bgcolor": bgcolor
                    })
        
        # Second pass: process each document with its metadata and relationships
        current_relation_type = None
        parent_stack = [None, None, None]  # Stack to track parents at each level
        
        for i, doc_info in enumerate(document_rows):
            row = doc_info["row"]
            doc_id = doc_info["doc_id"]
            level = doc_info["level"]
            bgcolor = doc_info["bgcolor"]
            
            # Extract JoOpen parameters
            detail_link = row.find("a", href=lambda href: href and "JoOpen" in href)
            jo_year, jo_number, jo_page, jo_lang = None, None, None, None
            if detail_link and "href" in detail_link.attrs:
                jo_match = re.search(r'JoOpen\("([^"]+)","([^"]+)","([^"]+)","([^"]+)"\)', detail_link["href"])
                if jo_match:
                    jo_year = jo_match.group(1)
                    jo_number = jo_match.group(2)
                    jo_page = jo_match.group(3)
                    jo_lang = jo_match.group(4)
            
            # Determine parent based on level
            parent_id = None
            if level > 0 and level < len(parent_stack):
                parent_id = parent_stack[level - 1]
            
            # Update parent stack
            parent_stack[level] = doc_id
            # Clear higher levels
            for l in range(level + 1, len(parent_stack)):
                parent_stack[l] = None
            
            # Create document
            doc = LegalDocument(
                doc_id=doc_id,
                doc_type="",
                level=level,
                parent_id=parent_id,
                jo_year=jo_year,
                jo_number=jo_number,
                jo_page=jo_page,
                jo_lang=jo_lang,
                bgcolor=bgcolor
            )
            
            # Find metadata rows for this document
            # They are the rows after this document header and before the next document header
            next_doc_row = None
            if i < len(document_rows) - 1:
                next_doc_row = document_rows[i + 1]["row"]
            
            # Process metadata rows
            current_row = row.find_next_sibling("tr")
            while current_row and (next_doc_row is None or current_row != next_doc_row):
                # Check for relationship indicator
                relation_font = current_row.find("font", {"color": "black"})
                if relation_font:
                    relation_text = relation_font.get_text(strip=True)
                    if "Abrog" in relation_text:
                        current_relation_type = "abrogated_by"
                    elif "Modifi" in relation_text:
                        current_relation_type = "modified_by"
                    elif "Rectificatif" in relation_text:
                        current_relation_type = "has_rectification"
                    elif "Texte(s) d'application" in relation_text:
                        current_relation_type = "implemented_by"
                    current_row = current_row.find_next_sibling("tr")
                    continue
                
                # Extract metadata
                cols = current_row.find_all("td")
                if len(cols) >= 1:
                    text = cols[-1].get_text(strip=True)
                    
                    # Document type and number
                    if doc.doc_type == "" and any(
                        doc_type in text 
                        for doc_type in ["Décret", "Arrêté", "Ordonnance", "Loi"]
                    ):
                        parts = text.split()
                        if len(parts) >= 3:
                            doc_type = " ".join(parts[:-2])
                            doc_number = parts[-2] if "n°" in parts[-2] else None
                            doc.doc_type = doc_type
                            doc.doc_number = doc_number
                            doc.date = parts[-1]
                    
                    # Ministry
                    elif "MINISTERE" in text or "MINISTRE" in text:
                        doc.ministry = text
                    
                    # Journal reference
                    elif "JO N°" in text:
                        match = re.search(r"JO N° (\d+) du (\d+ \w+ \d+), Page (\d+)", text)
                        if match:
                            doc.journal_number = match.group(1)
                            doc.journal_date = match.group(2)
                            doc.journal_page = match.group(3)
                    
                    # Document title
                    elif cols[-1].find("font", {"color": "#808080"}):
                        doc.title = text
                
                current_row = current_row.find_next_sibling("tr")
            
            # Add document to collection
            documents.append(doc)
            self.documents[doc_id] = doc
        
        # Third pass: establish relationships between documents
        for doc_id, doc in self.documents.items():
            if doc.parent_id:
                parent_doc = self.documents.get(doc.parent_id)
                if parent_doc:
                    # Find the relationship rows between parent and child
                    parent_row = None
                    for doc_info in document_rows:
                        if doc_info["doc_id"] == doc.parent_id:
                            parent_row = doc_info["row"]
                            break
                    
                    child_row = None
                    for doc_info in document_rows:
                        if doc_info["doc_id"] == doc_id:
                            child_row = doc_info["row"]
                            break
                    
                    if parent_row and child_row:
                        # Look for relationship indicator between parent and child
                        current_row = parent_row.find_next_sibling("tr")
                        relation_type = None
                        
                        while current_row and current_row != child_row:
                            relation_font = current_row.find("font", {"color": "black"})
                            if relation_font:
                                relation_text = relation_font.get_text(strip=True)
                                if "Abrog" in relation_text:
                                    relation_type = "abrogated_by"
                                elif "Modifi" in relation_text:
                                    relation_type = "modified_by"
                                elif "Rectificatif" in relation_text:
                                    relation_type = "has_rectification"
                                elif "Texte(s) d'application" in relation_text:
                                    relation_type = "implemented_by"
                                break
                            
                            current_row = current_row.find_next_sibling("tr")
                        
                        # Set up relationship
                        if relation_type:
                            if relation_type == "abrogated_by":
                                parent_doc.abrogated_by.append(doc_id)
                                doc.abrogates.append(doc.parent_id)
                            elif relation_type == "modified_by":
                                parent_doc.modified_by.append(doc_id)
                                doc.modifies.append(doc.parent_id)
                            elif relation_type == "has_rectification":
                                parent_doc.has_rectification.append(doc_id)
                                doc.is_rectification_of.append(doc.parent_id)
                            elif relation_type == "implemented_by":
                                parent_doc.implemented_by.append(doc_id)
                                doc.implements.append(doc.parent_id)

        logger.info(f"Extracted {len(documents)} documents from the current page")
        return documents

    def scrape_multiple_pages(
        self, ministry: str, start_page: int = 1, end_page: int = 5
    ) -> Dict[str, LegalDocument]:
        """
        Scrape multiple pages of search results.

        Args:
            ministry: Ministry/department to filter by
            start_page: First page to scrape
            end_page: Last page to scrape

        Returns:
            Dictionary of document ID to LegalDocument
        """
        # First page using search
        html = self.search(ministry=ministry)
        self.parse_html(html)
        
        # Subsequent pages using pagination
        for page in range(start_page + 1, end_page + 1):
            try:
                html = self.paginate(page)
                self.parse_html(html)
            except Exception as e:
                logger.error(f"Error scraping page {page}: {e}")
                break
                
        return self.documents

    def save_to_json(self, output_file: str) -> None:
        """
        Save the extracted documents to a JSON file.

        Args:
            output_file: Path to the output JSON file
        """
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {doc_id: doc.to_dict() for doc_id, doc in self.documents.items()},
                f,
                ensure_ascii=False,
                indent=2,
            )
        logger.info(f"Saved {len(self.documents)} documents to {output_file}")

    def load_from_json(self, input_file: str) -> None:
        """
        Load documents from a JSON file.

        Args:
            input_file: Path to the input JSON file
        """
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        self.documents = {}
        for doc_id, doc_data in data.items():
            self.documents[doc_id] = LegalDocument(**doc_data)
            
        logger.info(f"Loaded {len(self.documents)} documents from {input_file}")

    def create_graph(self) -> nx.DiGraph:
        """
        Create a directed graph of legal documents and their relationships.

        Returns:
            NetworkX DiGraph object
        """
        G = nx.DiGraph()
        
        # Add nodes
        for doc_id, doc in self.documents.items():
            # Create a label with the most important information
            label = f"{doc.doc_type} {doc.doc_number or ''}\n{doc.date or ''}"
            if doc.jo_year and doc.jo_number:
                label += f"\nJO {doc.jo_number}/{doc.jo_year}"
            
            # Determine node color based on document type or bgcolor
            if doc.bgcolor:
                color = doc.bgcolor
            elif "Décret" in doc.doc_type:
                color = "lightblue"
            elif "Arrêté" in doc.doc_type:
                color = "lightgreen"
            elif "Ordonnance" in doc.doc_type:
                color = "orange"
            elif "Loi" in doc.doc_type:
                color = "pink"
            else:
                color = "gray"
                
            G.add_node(
                doc_id,
                label=label,
                title=doc.title or "",
                ministry=doc.ministry or "",
                color=color,
                doc_type=doc.doc_type,
                level=doc.level,
                jo_year=doc.jo_year,
                jo_number=doc.jo_number,
                jo_page=doc.jo_page,
            )
        
        # Add parent-child relationships (hierarchical)
        for doc_id, doc in self.documents.items():
            if doc.parent_id and doc.parent_id in self.documents:
                G.add_edge(
                    doc.parent_id, 
                    doc_id, 
                    relationship="parent_child", 
                    color="black", 
                    style="dashed"
                )
        
        # Add other relationships
        for doc_id, doc in self.documents.items():
            # Abrogation relationships
            for related_id in doc.abrogates:
                if related_id in self.documents:
                    G.add_edge(doc_id, related_id, relationship="abrogates", color="red")
            
            # Modification relationships
            for related_id in doc.modifies:
                if related_id in self.documents:
                    G.add_edge(doc_id, related_id, relationship="modifies", color="blue")
            
            # Implementation relationships
            for related_id in doc.implements:
                if related_id in self.documents:
                    G.add_edge(doc_id, related_id, relationship="implements", color="green")
            
            # Rectification relationships
            for related_id in doc.is_rectification_of:
                if related_id in self.documents:
                    G.add_edge(doc_id, related_id, relationship="rectifies", color="purple")
        
        return G

    def visualize_graph(
        self, output_file: Optional[str] = None, show: bool = True
    ) -> None:
        """
        Visualize the legal document graph.

        Args:
            output_file: Path to save the visualization (optional)
            show: Whether to display the visualization
        """
        G = self.create_graph()
        
        if len(G) == 0:
            logger.warning("No documents to visualize")
            return
            
        logger.info(f"Visualizing graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
        
        # Set up the figure
        plt.figure(figsize=(15, 10))
        
        # Define node colors
        node_colors = [G.nodes[n]["color"] for n in G.nodes]
        
        # Define edge colors and styles
        edge_colors = []
        edge_styles = []
        for u, v, data in G.edges(data=True):
            edge_colors.append(data.get("color", "black"))
            edge_styles.append("dashed" if data.get("relationship") == "parent_child" else "solid")
        
        # Use a hierarchical layout
        try:
            # Try to use a hierarchical layout
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot", args="-Grankdir=TB")
        except:
            # Fall back to spring layout
            pos = nx.spring_layout(G, seed=42)
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, alpha=0.8, node_size=500)
        
        # Draw edges with different styles
        solid_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("relationship") != "parent_child"]
        dashed_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("relationship") == "parent_child"]
        
        nx.draw_networkx_edges(
            G, pos, edgelist=solid_edges, 
            edge_color=[G.edges[e]["color"] for e in solid_edges], 
            width=1.5, alpha=0.7, style="solid"
        )
        
        nx.draw_networkx_edges(
            G, pos, edgelist=dashed_edges, 
            edge_color=[G.edges[e]["color"] for e in dashed_edges], 
            width=1.0, alpha=0.5, style="dashed"
        )
        
        # Add labels with smaller font
        labels = {n: G.nodes[n]["label"] for n in G.nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
        
        # Add a legend for relationships
        legend_elements = [
            plt.Line2D([0], [0], color="red", lw=2, label="Abrogates"),
            plt.Line2D([0], [0], color="blue", lw=2, label="Modifies"),
            plt.Line2D([0], [0], color="green", lw=2, label="Implements"),
            plt.Line2D([0], [0], color="purple", lw=2, label="Rectifies"),
            plt.Line2D([0], [0], color="black", lw=2, linestyle="dashed", label="Hierarchical"),
        ]
        plt.legend(handles=legend_elements, loc="upper right")
        
        plt.title("Legal Document Relationships")
        plt.axis("off")
        
        # Save the figure if output file is specified
        if output_file:
            plt.savefig(output_file, bbox_inches="tight", dpi=300)
            logger.info(f"Saved visualization to {output_file}")
        
        # Show the figure if requested
        if show:
            plt.show()
        
        plt.close()

    def export_to_graphml(self, output_file: str) -> None:
        """
        Export the graph to GraphML format for use in other tools.

        Args:
            output_file: Path to save the GraphML file
        """
        G = self.create_graph()
        nx.write_graphml(G, output_file)
        logger.info(f"Exported graph to GraphML: {output_file}")


def main():
    """Main entry point for the JORADP scraper."""
    parser = argparse.ArgumentParser(description="Scrape and analyze JORADP legal documents")
    parser.add_argument(
        "--ministry", 
        type=str, 
        default="AFFAIRES RELIGIEUSES",
        help="Ministry to search for (default: AFFAIRES RELIGIEUSES)"
    )
    parser.add_argument(
        "--pages", 
        type=int, 
        default=3,
        help="Number of pages to scrape (default: 3)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="joradp_documents.json",
        help="Output JSON file (default: joradp_documents.json)"
    )
    parser.add_argument(
        "--visualize", 
        action="store_true",
        help="Visualize the document graph"
    )
    parser.add_argument(
        "--load", 
        type=str,
        help="Load documents from JSON file instead of scraping"
    )
    parser.add_argument(
        "--graph-output", 
        type=str,
        help="Save visualization to file"
    )
    parser.add_argument(
        "--graphml", 
        type=str,
        help="Export graph to GraphML file"
    )
    
    args = parser.parse_args()
    
    scraper = JoradpScraper()
    
    if args.load:
        # Load documents from JSON file
        scraper.load_from_json(args.load)
    else:
        # Scrape documents from website
        scraper.scrape_multiple_pages(
            ministry=args.ministry,
            start_page=1,
            end_page=args.pages
        )
        # Save to JSON
        scraper.save_to_json(args.output)
    
    if args.visualize or args.graph_output:
        # Visualize the graph
        scraper.visualize_graph(
            output_file=args.graph_output,
            show=args.visualize
        )
    
    if args.graphml:
        # Export to GraphML
        scraper.export_to_graphml(args.graphml)


if __name__ == "__main__":
    main()
