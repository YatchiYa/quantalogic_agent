"""Knowledge Graph Enhanced RAG Tool using LlamaIndex.

This tool combines traditional RAG with knowledge graph capabilities for enhanced
context understanding and relationship-aware retrieval.
"""

import os
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import json
import shutil

import chromadb
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    Document,
)
from llama_index.graph_stores.nebula import NebulaGraphStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.indices.knowledge_graph import KnowledgeGraphIndex
from llama_index.core.query_engine import KnowledgeGraphQueryEngine
from loguru import logger
from quantalogic.tools.tool import Tool, ToolArgument

# Configure tool-specific logging
logger.remove()
logger.add(
    sink=lambda msg: print(msg, end=""),
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
)

@dataclass
class KGNode:
    """Knowledge Graph node representation."""
    id: str
    type: str
    properties: Dict[str, Any]

@dataclass
class KGRelation:
    """Knowledge Graph relation representation."""
    source: str
    target: str
    relation_type: str
    properties: Dict[str, Any]

class KnowledgeGraphRAG(Tool):
    """Knowledge Graph enhanced RAG tool using LlamaIndex."""

    name: str = "kg_rag"
    description: str = (
        "Knowledge Graph enhanced RAG tool that combines traditional document retrieval "
        "with graph-based knowledge representation for improved context understanding."
    )
    arguments: List[ToolArgument] = [
        ToolArgument(
            name="query",
            arg_type="string",
            description="Query to search for relevant information in the knowledge graph and documents",
            required=True,
            example="What are the legal procedures for property disputes?",
        ),
        ToolArgument(
            name="max_sources",
            arg_type="int",
            description="Maximum number of sources to return",
            required=False,
            default="5",
        ),
        ToolArgument(
            name="min_relevance",
            arg_type="float",
            description="Minimum relevance score (0-1) for returned sources",
            required=False,
            default="0.1",
        ),
    ]

    def __init__(
        self,
        name: str = "kg_rag",
        persist_dir: str = "./storage/kg_rag",
        document_paths: Optional[List[str]] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        force_reindex: bool = False,
        model_name: str = "text-embedding-3-large",
        extract_triplets: bool = True
    ):
        """Initialize the Knowledge Graph RAG tool.
        
        Args:
            name: Name of the tool instance
            persist_dir: Directory to persist indices and graph
            document_paths: List of paths to documents to index
            chunk_size: Size of text chunks for indexing
            chunk_overlap: Overlap between chunks
            force_reindex: Whether to force reindexing
            model_name: OpenAI embedding model name
            extract_triplets: Whether to extract knowledge triplets from text
        """
        super().__init__()
        self.name = name
        self.persist_dir = os.path.abspath(persist_dir)
        self.force_reindex = force_reindex
        self.extract_triplets = extract_triplets

        logger.info("Initializing KnowledgeGraphRAG with parameters:")
        logger.info(f"  Name: {name}")
        logger.info(f"  Persist Directory: {self.persist_dir}")
        logger.info(f"  Document Paths: {document_paths}")
        logger.info(f"  Chunk Size: {chunk_size}")
        logger.info(f"  Chunk Overlap: {chunk_overlap}")
        logger.info(f"  Force Reindex: {force_reindex}")
        logger.info(f"  Model Name: {model_name}")
        logger.info(f"  Extract Triplets: {extract_triplets}")

        # Initialize embedding model
        try:
            logger.info(f"=> Initializing OpenAI embedding model ({model_name})...")
            self.embed_model = OpenAIEmbedding(
                model_name=model_name,
                embed_batch_size=8
            )
            logger.success(f"Successfully initialized OpenAI embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise

        # Setup vector store
        logger.info("=> Setting up vector store...")
        self._setup_vector_store()
        
        # Setup knowledge graph store
        logger.info("=> Setting up knowledge graph store...")
        self._setup_graph_store()
        
        # Initialize text splitter
        self.text_splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Initialize indices
        self.vector_index = None
        self.kg_index = None
        self._initialize_indices(document_paths)

    def _setup_vector_store(self):
        """Setup ChromaDB vector store."""
        chroma_persist_dir = os.path.join(self.persist_dir, "chroma")
        if self.force_reindex and os.path.exists(chroma_persist_dir):
            shutil.rmtree(chroma_persist_dir)
        
        os.makedirs(chroma_persist_dir, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=chroma_persist_dir)
        collection = chroma_client.create_collection(
            name="kg_document_collection",
            get_or_create=True
        )
        
        self.vector_store = ChromaVectorStore(chroma_collection=collection)
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

    def _setup_graph_store(self):
        """Setup knowledge graph store."""
        graph_persist_dir = os.path.join(self.persist_dir, "graph")
        os.makedirs(graph_persist_dir, exist_ok=True)
        
        # Initialize graph store (using simple file-based store for now)
        self.graph_store = NebulaGraphStore(
            space_name="knowledge_space",
            username="root",
            password="nebula",
            address="localhost",
            port=9669
        )

    def _extract_knowledge_triplets(self, text: str) -> List[Dict]:
        """Extract knowledge triplets from text using LLM.
        
        This uses a structured output LLM call to extract subject-predicate-object
        relationships from the text.
        """
        # TODO: Implement triplet extraction using LLM
        # For now return empty list
        return []

    def _load_documents(self, document_paths: List[str]) -> List[Document]:
        """Load and preprocess documents."""
        all_documents = []
        
        for path in document_paths:
            if not os.path.exists(path):
                logger.warning(f"Document path does not exist: {path}")
                continue
                
            try:
                docs = SimpleDirectoryReader(
                    input_files=[path],
                    filename_as_id=True
                ).load_data()
                
                for doc in docs:
                    # Extract knowledge triplets if enabled
                    if self.extract_triplets:
                        triplets = self._extract_knowledge_triplets(doc.text)
                        doc.metadata["knowledge_triplets"] = triplets
                    
                    all_documents.append(doc)
                    
            except Exception as e:
                logger.error(f"Error loading document {path}: {str(e)}")
                continue
                
        return all_documents

    def _initialize_indices(self, document_paths: Optional[List[str]]):
        """Initialize or load vector and knowledge graph indices."""
        if not document_paths and not self.force_reindex:
            try:
                # Try loading existing indices
                self.vector_index = load_index_from_storage(
                    storage_context=self.storage_context
                )
                self.kg_index = KnowledgeGraphIndex(
                    [],
                    storage_context=self.storage_context,
                    kg_store=self.graph_store
                )
                return
            except Exception as e:
                logger.error(f"Failed to load existing indices: {e}")
        
        if document_paths:
            documents = self._load_documents(document_paths)
            if documents:
                # Create vector index
                self.vector_index = VectorStoreIndex.from_documents(
                    documents,
                    storage_context=self.storage_context,
                    transformations=[self.text_splitter],
                    show_progress=True
                )
                
                # Create knowledge graph index
                self.kg_index = KnowledgeGraphIndex(
                    documents,
                    storage_context=self.storage_context,
                    kg_store=self.graph_store,
                    max_triplets_per_chunk=10,
                    include_embeddings=True,
                )
                
                # Persist indices
                self.storage_context.persist(persist_dir=self.persist_dir)
                logger.info(f"Created and persisted indices with {len(documents)} documents")

    def execute(
        self,
        query: str,
        max_sources: int = 5,
        min_relevance: float = 0.1
    ) -> str:
        """Execute search with the provided query.
        
        This combines results from both vector search and knowledge graph
        traversal for enhanced retrieval.
        """
        try:
            if not self.vector_index or not self.kg_index:
                raise ValueError("No indices available. Please add documents first.")

            logger.info(f"Processing query: {query}")
            
            # Get results from vector index
            vector_engine = self.vector_index.as_query_engine(
                similarity_top_k=max_sources,
                node_postprocessors=[
                    SimilarityPostprocessor(similarity_cutoff=min_relevance)
                ],
                response_mode="no_text",
                verbose=True
            )
            
            # Get results from knowledge graph
            kg_engine = KnowledgeGraphQueryEngine(
                index=self.kg_index,
                max_hops=2,
                include_text=True,
                response_mode="tree",
                verbose=True
            )
            
            # Execute queries
            vector_response = vector_engine.query(query)
            kg_response = kg_engine.query(query)
            
            # Process vector results
            vector_results = []
            for node in vector_response.source_nodes:
                if node.score < min_relevance:
                    continue
                    
                result = {
                    'content': node.node.text.strip(),
                    'score': round(float(node.score), 4),
                    'metadata': node.node.metadata,
                    'source': 'vector'
                }
                vector_results.append(result)
            
            # Process knowledge graph results
            kg_results = []
            if hasattr(kg_response, 'source_nodes'):
                for node in kg_response.source_nodes:
                    result = {
                        'content': node.node.text.strip(),
                        'score': getattr(node, 'score', 1.0),
                        'metadata': node.node.metadata,
                        'source': 'knowledge_graph'
                    }
                    kg_results.append(result)
            
            # Combine and sort results
            all_results = vector_results + kg_results
            sorted_results = sorted(
                all_results,
                key=lambda x: x['score'],
                reverse=True
            )[:max_sources]
            
            # Prepare response
            response = {
                'query': query,
                'sources': sorted_results,
                'vector_sources': len(vector_results),
                'kg_sources': len(kg_results),
                'timestamp': datetime.now().isoformat()
            }
            
            return json.dumps(response, indent=4, ensure_ascii=False)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Search failed: {error_msg}")
            error_response = {
                'error': error_msg,
                'query': query,
                'timestamp': datetime.now().isoformat()
            }
            return json.dumps(error_response, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    # Example usage
    try:
        # Initialize tool
        tool = KnowledgeGraphRAG(
            persist_dir="./storage/kg_rag",
            document_paths=[
                "./docs/folder_test/code_civile.md",
                "./docs/folder_test/code_procedure.md"
            ],
            chunk_size=512,
            chunk_overlap=128,
            force_reindex=True,
            model_name="text-embedding-3-large",
            extract_triplets=True
        )
        
        # Test query
        test_query = "Quels sont les recours légaux en Algérie pour contraindre un voisin à fermer des ouvertures (fenêtres) donnant sur ma propriété ?"
        
        result = tool.execute(
            query=test_query,
            max_sources=5,
            min_relevance=0.1
        )
        print(result)
        
    except Exception as e:
        logger.error(f"Failed to initialize or run tool: {e}")
