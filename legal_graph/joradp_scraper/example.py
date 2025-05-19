#!/usr/bin/env python3
"""
Example script demonstrating how to use the JORADP scraper.
This script performs a small test scrape and visualizes the results.
"""

from joradp_scraper import JoradpScraper
from loguru import logger
import matplotlib.pyplot as plt
import networkx as nx

def run_example():
    """Run a demonstration of the JORADP scraper."""
    # Create a scraper instance
    scraper = JoradpScraper()
    
    # Option 1: Scrape live data (limited to 2 pages for demo to capture more relationships)
    logger.info("Scraping 2 pages of documents from JORADP...")
    scraper.scrape_multiple_pages(
        ministry="AFFAIRES RELIGIEUSES",
        start_page=1,
        end_page=2
    )
    
    # Save the scraped data
    output_file = "example_documents.json"
    scraper.save_to_json(output_file)
    logger.info(f"Saved {len(scraper.documents)} documents to {output_file}")
    
    # Visualize the document relationships
    logger.info("Generating visualization...")
    scraper.visualize_graph(output_file="example_graph.png", show=True)
    
    # Export to GraphML for use in other tools
    scraper.export_to_graphml("example_graph.graphml")
    
    # Print some statistics
    doc_types = {}
    for doc in scraper.documents.values():
        doc_type = doc.doc_type.split()[0] if doc.doc_type else "Unknown"
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
    
    logger.info("Document type statistics:")
    for doc_type, count in sorted(doc_types.items()):
        logger.info(f"  {doc_type}: {count}")
    
    # Count documents by level
    levels = {}
    for doc in scraper.documents.values():
        levels[doc.level] = levels.get(doc.level, 0) + 1
    
    logger.info("Document level statistics:")
    for level, count in sorted(levels.items()):
        logger.info(f"  Level {level}: {count}")
    
    # Find documents with JoOpen parameters
    docs_with_joopen = [doc for doc in scraper.documents.values() 
                        if doc.jo_year and doc.jo_number]
    
    logger.info(f"Documents with JoOpen parameters: {len(docs_with_joopen)}")
    
    if docs_with_joopen:
        logger.info("Sample JoOpen parameters:")
        for i, doc in enumerate(docs_with_joopen[:5]):
            logger.info(f"  {i+1}. {doc.doc_type} {doc.doc_number or ''} - JO {doc.jo_number}/{doc.jo_year}, Page {doc.jo_page}")
    
    # Find documents with hierarchical relationships
    docs_with_parent = [doc for doc in scraper.documents.values() if doc.parent_id]
    
    logger.info(f"Documents with parent-child relationships: {len(docs_with_parent)}")
    
    if docs_with_parent:
        logger.info("Sample hierarchical relationships:")
        for i, doc in enumerate(docs_with_parent[:5]):
            parent = scraper.documents.get(doc.parent_id)
            parent_desc = f"{parent.doc_type} {parent.doc_number or ''}" if parent else "Unknown"
            logger.info(f"  {i+1}. {doc.doc_type} {doc.doc_number or ''} (Level {doc.level}) -> Parent: {parent_desc}")
    
    # Create a small visualization of just the hierarchical relationships
    if docs_with_parent:
        logger.info("Creating a focused visualization of hierarchical relationships...")
        # Create a subgraph with just the hierarchical relationships
        G = nx.DiGraph()
        
        # Add nodes for documents with parent-child relationships
        doc_ids = set()
        for doc in docs_with_parent:
            doc_ids.add(doc.doc_id)
            if doc.parent_id:
                doc_ids.add(doc.parent_id)
        
        for doc_id in doc_ids:
            doc = scraper.documents.get(doc_id)
            if not doc:
                continue
                
            label = f"{doc.doc_type} {doc.doc_number or ''}"
            G.add_node(doc_id, label=label, level=doc.level)
        
        # Add edges for parent-child relationships
        for doc in docs_with_parent:
            if doc.parent_id in doc_ids:
                G.add_edge(doc.parent_id, doc.doc_id)
        
        # Create a hierarchical layout
        plt.figure(figsize=(12, 8))
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot", args="-Grankdir=TB")
        except:
            pos = nx.spring_layout(G, seed=42)
            
        # Color nodes by level
        node_colors = ["#78a7b9" if G.nodes[n].get("level") == 0 else 
                       "#9ec7d7" if G.nodes[n].get("level") == 1 else 
                       "#c8e7f3" for n in G.nodes]
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, alpha=0.8, node_size=300)
        nx.draw_networkx_edges(G, pos, edge_color="black", width=1.0, alpha=0.5, style="dashed")
        nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]["label"] for n in G.nodes}, font_size=8)
        
        plt.title("Hierarchical Document Relationships")
        plt.axis("off")
        plt.savefig("hierarchical_relationships.png", bbox_inches="tight", dpi=300)
        logger.info("Saved hierarchical visualization to hierarchical_relationships.png")

if __name__ == "__main__":
    run_example()
