#!/usr/bin/env python3
"""
Test script for parsing local HTML data with hierarchical relationships.
"""

from joradp_scraper import JoradpScraper
from loguru import logger
import matplotlib.pyplot as plt
import networkx as nx

def test_local_html():
    """Test parsing of local HTML file with hierarchical relationships."""
    # Create a scraper instance
    scraper = JoradpScraper()
    
    # Load test HTML data
    logger.info("Loading test HTML data...")
    with open("test_data.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    
    # Parse the HTML
    documents = scraper.parse_html(html_content)
    
    # Save the parsed data
    output_file = "test_documents.json"
    scraper.save_to_json(output_file)
    logger.info(f"Saved {len(scraper.documents)} documents to {output_file}")
    
    # Print document information
    for doc_id, doc in scraper.documents.items():
        logger.info(f"Document ID: {doc_id}, Type: {doc.doc_type}, Level: {doc.level}")
        if doc.parent_id:
            logger.info(f"  Parent ID: {doc.parent_id}")
        if doc.abrogates:
            logger.info(f"  Abrogates: {doc.abrogates}")
        if doc.abrogated_by:
            logger.info(f"  Abrogated by: {doc.abrogated_by}")
        if doc.modifies:
            logger.info(f"  Modifies: {doc.modifies}")
        if doc.modified_by:
            logger.info(f"  Modified by: {doc.modified_by}")
    
    # Visualize the document relationships
    logger.info("Generating visualization...")
    scraper.visualize_graph(output_file="test_graph.png", show=False)
    
    # Create a focused visualization of just the hierarchical relationships
    logger.info("Creating a focused visualization of hierarchical relationships...")
    G = nx.DiGraph()
    
    # Add nodes
    for doc_id, doc in scraper.documents.items():
        label = f"{doc.doc_type}\n{doc.doc_id}"
        G.add_node(
            doc_id, 
            label=label, 
            level=doc.level,
            color="#78a7b9" if doc.level == 0 else "#9ec7d7" if doc.level == 1 else "#c8e7f3"
        )
    
    # Add parent-child edges
    for doc_id, doc in scraper.documents.items():
        if doc.parent_id:
            G.add_edge(
                doc.parent_id, 
                doc_id, 
                relationship="parent_child",
                color="black",
                style="dashed"
            )
    
    # Add relationship edges
    for doc_id, doc in scraper.documents.items():
        # Abrogation relationships
        for related_id in doc.abrogates:
            if related_id in scraper.documents:
                G.add_edge(
                    doc_id, 
                    related_id, 
                    relationship="abrogates",
                    color="red",
                    style="solid"
                )
        
        # Modification relationships
        for related_id in doc.modifies:
            if related_id in scraper.documents:
                G.add_edge(
                    doc_id, 
                    related_id, 
                    relationship="modifies",
                    color="blue",
                    style="solid"
                )
    
    # Create a hierarchical layout
    plt.figure(figsize=(12, 8))
    try:
        # Try to use graphviz for better hierarchical layout
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot", args="-Grankdir=TB")
    except:
        # Fall back to spring layout
        pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes with colors based on level
    node_colors = [G.nodes[n]["color"] for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, alpha=0.8, node_size=2000)
    
    # Draw edges with different styles for different relationships
    edge_colors = []
    edge_styles = []
    
    # Separate edges by style
    dashed_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("style") == "dashed"]
    solid_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("style") == "solid"]
    
    # Draw dashed edges (parent-child)
    if dashed_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=dashed_edges,
            edge_color="black", width=1.0, alpha=0.7, style="dashed"
        )
    
    # Draw solid edges (relationships)
    if solid_edges:
        edge_colors = [G.edges[e]["color"] for e in solid_edges]
        nx.draw_networkx_edges(
            G, pos, edgelist=solid_edges,
            edge_color=edge_colors, width=1.5, alpha=0.7
        )
    
    # Add labels
    labels = {n: G.nodes[n]["label"] for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)
    
    # Add a legend
    legend_elements = [
        plt.Line2D([0], [0], color="black", linestyle="dashed", lw=1, label="Parent-Child"),
        plt.Line2D([0], [0], color="red", lw=1.5, label="Abrogates"),
        plt.Line2D([0], [0], color="blue", lw=1.5, label="Modifies")
    ]
    plt.legend(handles=legend_elements, loc="upper right")
    
    plt.title("Legal Document Hierarchical Relationships")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("test_hierarchical_graph.png", dpi=300, bbox_inches="tight")
    logger.info("Saved hierarchical visualization to test_hierarchical_graph.png")

if __name__ == "__main__":
    test_local_html()
