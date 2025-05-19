# JORADP Legal Document Scraper

A tool for scraping, parsing, and visualizing legal documents from the Algerian Official Journal (Journal Officiel de la République Algérienne).

## Features

- Scrapes legal documents from the JORADP website
- Extracts document metadata (type, number, date, ministry, etc.)
- Identifies relationships between documents (abrogations, modifications, etc.)
- Visualizes document relationships as a directed graph
- Exports data to JSON and GraphML formats

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Scrape 3 pages of documents from the Ministry of Religious Affairs
python joradp_scraper.py --ministry "AFFAIRES RELIGIEUSES" --pages 3

# Visualize the document graph
python joradp_scraper.py --load joradp_documents.json --visualize

# Export to GraphML for use in other tools
python joradp_scraper.py --load joradp_documents.json --graphml joradp_graph.graphml
```

### Command Line Arguments

- `--ministry`: Ministry to search for (default: "AFFAIRES RELIGIEUSES")
- `--pages`: Number of pages to scrape (default: 3)
- `--output`: Output JSON file (default: joradp_documents.json)
- `--visualize`: Visualize the document graph
- `--load`: Load documents from JSON file instead of scraping
- `--graph-output`: Save visualization to image file
- `--graphml`: Export graph to GraphML file

## Example

```bash
# Scrape documents and visualize the graph
python joradp_scraper.py --ministry "AFFAIRES RELIGIEUSES" --pages 5 --visualize --graph-output legal_graph.png
```

## Document Relationships

The scraper identifies several types of relationships between legal documents:

- **Abrogation**: A document that repeals or cancels another document
- **Modification**: A document that changes or amends another document
- **Implementation**: A document that implements another document
- **Rectification**: A document that corrects another document

These relationships are visualized in the graph with different colors.
