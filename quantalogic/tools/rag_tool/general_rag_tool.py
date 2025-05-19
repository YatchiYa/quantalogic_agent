"""General Purpose RAG Tool using OpenAI embeddings for versatile document retrieval and QA.

This tool provides core RAG capabilities including:
- Document processing and chunking
- OpenAI embeddings for semantic search
- Flexible document types support
- Enhanced context management
- Configurable response modes
"""

import os
from typing import List, Optional, Dict, Any, Literal
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime
import shutil

import chromadb
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    Document,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
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
class DocumentMetadata:
    """Structured metadata for documents."""
    title: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    source_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {k: v for k, v in self.__dict__.items() if v is not None}

@dataclass
class DocumentSource:
    """Structured representation of a document source."""
    content: str
    file_name: str
    metadata: Optional[DocumentMetadata] = None
    score: Optional[float] = None

class ResponseMode(str, Enum):
    """Response modes for the RAG tool."""
    SOURCES_ONLY = "sources_only"
    CONTEXTUAL_ANSWER = "contextual_answer"
    ANSWER_WITH_SOURCES = "answer_with_sources"

class GeneralRAG(Tool):
    """General-purpose RAG tool using OpenAI embeddings."""

    name: str = "general_rag"
    description: str = (
        "General-purpose RAG tool for document retrieval and QA using OpenAI embeddings. "
        "Supports various document types and flexible response modes."
    )
    arguments: List[ToolArgument] = [
        ToolArgument(
            name="query",
            arg_type="string",
            description="Query to search for relevant information in the indexed documents",
            required=True,
            example="What are the key features of the product?",
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
        ToolArgument(
            name="response_mode",
            arg_type="string",
            description="Response mode: 'sources_only', 'contextual_answer', or 'answer_with_sources'",
            required=False,
            default="sources_only",
        ),
    ]

    def __init__(
        self,
        name: str = "general_rag",
        persist_dir: str = "./storage/general_rag",
        document_paths: Optional[List[str]] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        force_reindex: bool = False,
        model_name: str = "text-embedding-ada-002",
        use_temp_dir: bool = True
    ):
        """Initialize the general RAG tool."""
        super().__init__()
        self.name = name
        self.use_temp_dir = use_temp_dir
        
        # Handle persistence directory
        if use_temp_dir:
            import tempfile
            self.temp_dir = tempfile.mkdtemp(prefix="chroma_")
            self.persist_dir = self.temp_dir
            logger.info(f"Using temporary directory: {self.temp_dir}")
        else:
            self.persist_dir = os.path.abspath(persist_dir)
            os.makedirs(self.persist_dir, exist_ok=True)
        
        self.force_reindex = force_reindex

        logger.info("Initializing GeneralRAG with parameters:")
        logger.info(f"  Name: {name}")
        logger.info(f"  Persist Directory: {self.persist_dir}")
        logger.info(f"  Document Paths: {document_paths}")
        logger.info(f"  Chunk Size: {chunk_size}")
        logger.info(f"  Chunk Overlap: {chunk_overlap}")
        logger.info(f"  Force Reindex: {force_reindex}")
        logger.info(f"  Model Name: {model_name}")
        logger.info(f"  Using Temp Dir: {use_temp_dir}")

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

        # Setup ChromaDB
        logger.info("=> Setting up ChromaDB...")
        chroma_persist_dir = os.path.join(self.persist_dir, "chroma")
        if force_reindex and os.path.exists(chroma_persist_dir):
            logger.warning(f"Force reindex enabled - removing existing ChromaDB at: {chroma_persist_dir}")
            try:
                shutil.rmtree(chroma_persist_dir)
            except Exception as e:
                logger.error(f"Failed to remove existing ChromaDB directory: {e}")
                raise RuntimeError(f"Cannot remove existing ChromaDB directory: {e}")
        
        try:
            # Create directory with proper permissions
            os.makedirs(chroma_persist_dir, exist_ok=True)
            
            # Initialize ChromaDB with settings
            chroma_settings = chromadb.Settings(
                allow_reset=True,
                is_persistent=True,
                persist_directory=chroma_persist_dir,
                anonymized_telemetry=False
            )
            
            chroma_client = chromadb.PersistentClient(
                path=chroma_persist_dir,
                settings=chroma_settings
            )
            
            # Create collection with proper settings
            collection = chroma_client.get_or_create_collection(
                name="document_collection",
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.success("Successfully initialized ChromaDB collection")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise RuntimeError(f"ChromaDB initialization failed: {e}")
        
        self.vector_store = ChromaVectorStore(chroma_collection=collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        logger.success("Successfully initialized vector store and storage context")
        
        # Initialize text splitter
        logger.info("=> Initializing text splitter...")
        self.text_splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        logger.success("Successfully initialized text splitter")
        
        # Initialize or load index
        logger.info("=> Initializing index...")
        self.index = self._initialize_index(document_paths)

    def __del__(self):
        """Cleanup temporary directory if used."""
        if hasattr(self, 'use_temp_dir') and self.use_temp_dir and hasattr(self, 'temp_dir'):
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                logger.error(f"Failed to cleanup temporary directory: {e}")

    def _extract_metadata(self, text: str, file_path: str) -> Dict[str, Any]:
        """Extract basic metadata from document."""
        metadata = {}
        
        # Extract filename and extension
        filename = os.path.basename(file_path)
        metadata["file_name"] = filename
        metadata["file_extension"] = os.path.splitext(filename)[1]
        
        # Get file stats
        stats = os.stat(file_path)
        metadata["file_size"] = stats.st_size
        metadata["last_modified"] = datetime.fromtimestamp(stats.st_mtime).isoformat()
        
        # Basic content stats
        metadata["char_count"] = len(text)
        metadata["word_count"] = len(text.split())
        
        return metadata

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
                
                # Process each document
                for doc in docs:
                    # Extract metadata
                    metadata = self._extract_metadata(doc.text, path)
                    metadata.update(doc.metadata)
                    
                    processed_doc = Document(
                        text=doc.text,
                        metadata=metadata
                    )
                    all_documents.append(processed_doc)
                    
            except Exception as e:
                logger.error(f"Error loading document {path}: {str(e)}")
                continue
                
        return all_documents

    def _check_index_exists(self) -> bool:
        """Check if index files exist and are valid."""
        required_files = [
            os.path.join(self.persist_dir, "docstore.json"),
            os.path.join(self.persist_dir, "chroma"),
            os.path.join(self.persist_dir, "chroma", "chroma.sqlite3")
        ]
        return all(os.path.exists(f) for f in required_files)

    def _initialize_index(self, document_paths: Optional[List[str]]) -> Optional[VectorStoreIndex]:
        """Initialize or load the vector index."""
        logger.info("Initializing index...")
        
        index_exists = self._check_index_exists()
        
        # Case 1: Force reindex requested
        if self.force_reindex:
            logger.info("Force reindex requested - creating new index")
            if document_paths:
                documents = self._load_documents(document_paths)
                return self._create_index(documents)
            else:
                logger.warning("Force reindex requested but no document paths provided")
                return None
        
        # Case 2: Index exists and no new documents
        if index_exists and not document_paths:
            logger.info("Loading existing index from storage")
            try:
                return load_index_from_storage(storage_context=self.storage_context)
            except Exception as e:
                logger.error(f"Failed to load existing index: {str(e)}")
                return None
        
        # Case 3: Index exists and new documents provided
        if index_exists and document_paths:
            logger.info("Loading existing index and updating with new documents")
            try:
                index = load_index_from_storage(storage_context=self.storage_context)
                new_documents = self._load_documents(document_paths)
                if new_documents:
                    for doc in new_documents:
                        index.insert(doc)
                    self.storage_context.persist(persist_dir=self.persist_dir)
                    logger.info(f"Updated index with {len(new_documents)} new documents")
                return index
            except Exception as e:
                logger.error(f"Failed to update existing index: {str(e)}")
                return None
        
        # Case 4: No index exists but documents provided
        if document_paths:
            logger.info("Creating new index from documents")
            documents = self._load_documents(document_paths)
            return self._create_index(documents)
        
        # Case 5: No index and no documents
        logger.warning("No index exists and no documents provided")
        return None

    def _create_index(self, documents: List[Document]) -> Optional[VectorStoreIndex]:
        """Create vector store index from documents."""
        try:
            if not documents:
                logger.warning("No valid documents provided")
                return None
                
            logger.info("Creating vector index...")
            
            # Ensure storage directory exists with proper permissions
            os.makedirs(self.persist_dir, exist_ok=True)
            
            # Create index with proper settings
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=self.storage_context,
                transformations=[self.text_splitter],
                show_progress=True
            )
            
            # Ensure storage directory is writable
            logger.info("Persisting index to storage...")
            try:
                self.storage_context.persist(persist_dir=self.persist_dir)
                logger.success(f"Created and persisted index with {len(documents)} documents")
            except Exception as e:
                logger.error(f"Failed to persist index: {e}")
                # Try to fix permissions and retry
                self.storage_context.persist(persist_dir=self.persist_dir)
                logger.success("Successfully persisted index after fixing permissions")
            
            return index
            
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            return None

    def _generate_contextual_answer(self, query: str, sources: List[Dict]) -> str:
        """Generate a contextual answer using the retrieved sources."""
        if not sources:
            return "No relevant sources found for your query."

        # Sort sources by score
        sorted_sources = sorted(sources, key=lambda x: x["score"], reverse=True)
        
        # Build context
        context_parts = []
        
        # Add summary of findings
        context_parts.append(f"Based on {len(sources)} relevant sources:")
        
        # Add most relevant content
        for i, source in enumerate(sorted_sources[:3], 1):
            context_parts.append(
                f"\n{i}. {source['content']}"
                f" (Source: {source['metadata']['file_name']}, "
                f"Relevance: {source['score']:.3f})"
            )

        # Add source references
        context_parts.append("\nSources:")
        for source in sorted_sources[:3]:
            context_parts.append(
                f"- {source['metadata']['file_name']}"
            )

        return "\n".join(context_parts)

    def execute(
        self, 
        query: str, 
        max_sources: int = 5, 
        min_relevance: float = 0.1,
        response_mode: str = ResponseMode.SOURCES_ONLY
    ) -> str:
        """Execute search with the provided query."""
        try:
            if not self.index:
                raise ValueError("No index available. Please add documents first.")

            logger.info(f"Processing query: {query}")
            
            # Get results
            query_engine = self.index.as_query_engine(
                similarity_top_k=max_sources,
                node_postprocessors=[
                    SimilarityPostprocessor(similarity_cutoff=min_relevance)
                ],
                response_mode="no_text",
                streaming=False,
                verbose=True
            )
            
            response = query_engine.query(query)
            
            # Process results
            results = []
            for node in response.source_nodes:
                if node.score < min_relevance:
                    continue
                
                metadata = node.node.metadata.copy()
                result = {
                    'content': node.node.text.strip(),
                    'score': round(float(node.score), 4),
                    'metadata': metadata
                }
                results.append(result)
            
            # Prepare response based on mode
            response_mode = ResponseMode(response_mode)
            if response_mode == ResponseMode.SOURCES_ONLY:
                return json.dumps({"sources": results, "query": query}, indent=4, ensure_ascii=False)
            elif response_mode == ResponseMode.CONTEXTUAL_ANSWER:
                answer = self._generate_contextual_answer(query, results)
                return json.dumps({"answer": answer, "query": query}, indent=4, ensure_ascii=False)
            else:  # ANSWER_WITH_SOURCES
                answer = self._generate_contextual_answer(query, results)
                return json.dumps({
                    "answer": answer,
                    "sources": results,
                    "query": query
                }, indent=4, ensure_ascii=False)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Search failed: {error_msg}")
            error_response = {
                'error': error_msg,
                'query': query,
                'timestamp': str(datetime.now().isoformat())
            }
            return json.dumps(error_response, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    # Example usage
    try:
        # Initialize tool
        tool = GeneralRAG(
            persist_dir="./storage/general_rag",
            document_paths=[
                "./docs/folder_test/code_civile.md",
                "./docs/folder_test/code_procedure.md"
            ],
            chunk_size=512,
            chunk_overlap=128,
            force_reindex=True,
            model_name="text-embedding-3-large",  # text-embedding-3-small # Specify OpenAI model
            use_temp_dir=False
        )
        
        # Test query
        test_query = "Quels sont les recours légaux en Algérie pour contraindre un voisin à fermer des ouvertures (fenêtres) donnant sur ma propriété ?"
        
        result = tool.execute(
            query=test_query,
            max_sources=5,
            min_relevance=0.1,
            response_mode=ResponseMode.ANSWER_WITH_SOURCES
        )
        print(result)
        
    except Exception as e:
        logger.error(f"Failed to initialize or run tool: {e}")