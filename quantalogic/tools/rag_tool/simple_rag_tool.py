"""Simple RAG Tool using LlamaIndex and ChromaDB for direct document retrieval and answers.

This tool provides streamlined RAG capabilities with direct text output:
- Document processing and chunking with configurable parameters
- Vector embeddings using OpenAI models
- ChromaDB for efficient vector storage
- Direct answer format for easy consumption
"""

from typing import Callable
import os
import shutil
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

import chromadb
import openai
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    Document,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
from loguru import logger
from pydantic import ConfigDict, Field

from quantalogic.console_print_token import console_print_token
from quantalogic.event_emitter import EventEmitter
from quantalogic.tools.tool import Tool, ToolArgument

# Configure tool-specific logging
logger.remove()
logger.add(
    sink=lambda msg: print(msg, end=""),
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
)

@dataclass
class DocumentSource:
    """Structured representation of a document source."""
    content: str
    file_name: str
    metadata: Optional[Dict[str, Any]] = None
    score: Optional[float] = None

class SimpleRagTool(Tool):
    """Simple RAG tool using LlamaIndex and ChromaDB with direct answer output."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = "simple_rag"
    description: str = (
        "Simple RAG tool for document retrieval and QA using LlamaIndex and ChromaDB. "
        "Returns direct text answers without JSON formatting."
    )
    arguments: List[ToolArgument] = [
        ToolArgument(
            name="query",
            arg_type="string",
            description="Query to search for relevant information in the indexed documents",
            required=True,
            example="What are the key provisions in the document?",
        ),
        ToolArgument(
            name="max_sources",
            arg_type="int",
            description="Maximum number of sources to include in the answer",
            required=False,
            default="8",
        ),
        ToolArgument(
            name="min_relevance",
            arg_type="float",
            description="Minimum relevance score (0-1) for included sources",
            required=False,
            default="0.1",
        ),
        ToolArgument(
            name="use_llm",
            arg_type="boolean",
            description="Whether to use LLM to generate a coherent answer",
            required=False,
            default="True",
        ),
    ]
    event_emitter: EventEmitter | None = Field(default=None, exclude=True)
    on_token: Callable | None = Field(default=None, exclude=True)

    def __init__(
        self,
        name: str = "simple_rag",
        persist_dir: str = None,
        document_paths: Optional[List[str]] = None,
        chunk_size: int = 1024,
        chunk_overlap: int = 256,
        force_reindex: bool = False,
        model_name: str = "text-embedding-3-large",
        use_temp_dir: bool = True,
        llm_model: str = "gpt-4o-mini",
        on_token: Callable = None,
        event_emitter: EventEmitter = None
    ):
        """Initialize the Simple RAG tool.
        
        Args:
            name: Name of the tool
            persist_dir: Directory to persist the vector store
            document_paths: List of document paths to index
            chunk_size: Size of text chunks for indexing
            chunk_overlap: Overlap between chunks
            force_reindex: Whether to force reindexing
            model_name: OpenAI embedding model name
            use_temp_dir: Whether to use a temporary directory
            llm_model: LLM model to use for answer generation
            on_token: Callback function for streaming tokens
            event_emitter: Event emitter for handling events
        """
        super().__init__()
        self.name = name
        self.use_temp_dir = use_temp_dir
        self.llm_model = llm_model
        self.on_token = on_token
        self.event_emitter = event_emitter
        
        # Handle persistence directory
        if use_temp_dir:
            import tempfile
            self.temp_dir = tempfile.mkdtemp(prefix="chroma_")
            self.persist_dir = self.temp_dir
            logger.info(f"Using temporary directory: {self.temp_dir}")
        else:
            # If no persist_dir is provided, use a directory in the user's home
            if persist_dir is None:
                home_dir = os.path.expanduser("~")
                persist_dir = os.path.join(home_dir, ".simple_rag_storage")
                logger.info(f"No persist_dir provided, using home directory: {persist_dir}")
            
            self.persist_dir = os.path.abspath(persist_dir)
            os.makedirs(self.persist_dir, exist_ok=True)
            
            # Try to set permissions
            try:
                os.chmod(self.persist_dir, 0o755)  # rwxr-xr-x
            except Exception as e:
                logger.warning(f"Could not set permissions on persist directory: {e}")
        
        self.force_reindex = force_reindex

        logger.info("Initializing SimpleRagTool with parameters:")
        logger.info(f"  Name: {name}")
        logger.info(f"  Persist Directory: {self.persist_dir}")
        logger.info(f"  Document Paths: {document_paths}")
        logger.info(f"  Chunk Size: {chunk_size}")
        logger.info(f"  Chunk Overlap: {chunk_overlap}")
        logger.info(f"  Force Reindex: {force_reindex}")
        logger.info(f"  Model Name: {model_name}")
        logger.info(f"  Using Temp Dir: {use_temp_dir}")
        logger.info(f"  LLM Model: {self.llm_model}")

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
            
            # Try with a fallback directory in user's home
            fallback_dir = os.path.join(os.path.expanduser("~"), ".simple_rag_storage", "chroma_db")
            logger.info(f"Attempting fallback to user home directory: {fallback_dir}")
            
            # Ensure the fallback directory exists and is writable
            os.makedirs(fallback_dir, exist_ok=True)
            try:
                os.chmod(fallback_dir, 0o755)  # rwxr-xr-x
            except Exception as e:
                logger.warning(f"Could not set permissions on fallback directory: {e}")
            
            # Try with the fallback directory
            chroma_client = chromadb.PersistentClient(path=fallback_dir)
            collection = chroma_client.get_or_create_collection(name="document_collection")
            self.vector_store = ChromaVectorStore(chroma_collection=collection)
            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            logger.success("Successfully created alternative storage context")
        
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
        """Extract basic metadata from document.
        
        Args:
            text: Document text
            file_path: Path to the document file
            
        Returns:
            Dictionary of metadata
        """
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
        
        # Try to extract document title and section info from text
        lines = text.strip().split('\n')
        if lines:
            # First non-empty line might be a title
            for line in lines:
                if line.strip():
                    metadata["title"] = line.strip()
                    break
                    
            # Try to identify sections
            sections = []
            for i, line in enumerate(lines):
                if line.strip().lower().startswith(('chapter', 'section', 'part', 'article')):
                    sections.append(line.strip())
            
            if sections:
                metadata["sections"] = sections
                
            # Try to extract page numbers
            page_info = None
            for line in lines:
                if "page" in line.lower():
                    page_info = line.strip()
                    break
            
            if page_info:
                metadata["page_info"] = page_info
        
        return metadata

    def _load_documents(self, document_paths: List[str]) -> List[Document]:
        """Load and preprocess documents.
        
        Args:
            document_paths: List of paths to documents
            
        Returns:
            List of processed Document objects
        """
        all_documents = []
        
        for path in document_paths:
            if not os.path.exists(path):
                logger.warning(f"Document path does not exist: {path}")
                continue
                
            try:
                logger.info(f"Loading document: {path}")
                docs = SimpleDirectoryReader(
                    input_files=[path],
                    filename_as_id=True
                ).load_data()
                
                logger.info(f"Loaded {len(docs)} document(s) from {path}")
                
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
        
        # Store documents for potential fallback use
        self.temp_documents = all_documents
                
        return all_documents

    def _check_index_exists(self) -> bool:
        """Check if index files exist and are valid.
        
        Returns:
            Boolean indicating if index exists
        """
        required_files = [
            os.path.join(self.persist_dir, "docstore.json"),
            os.path.join(self.persist_dir, "chroma"),
            os.path.join(self.persist_dir, "chroma", "chroma.sqlite3")
        ]
        return all(os.path.exists(f) for f in required_files)

    def _initialize_index(self, document_paths: Optional[List[str]]) -> Optional[VectorStoreIndex]:
        """Initialize or load the vector index.
        
        Args:
            document_paths: List of document paths to index
            
        Returns:
            Vector store index or None if initialization fails
        """
        logger.info("Initializing index...")
        
        # Try to use the home directory if we encounter permission issues
        def try_home_directory_fallback():
            home_dir = os.path.expanduser("~")
            alt_dir = os.path.join(home_dir, ".simple_rag_storage")
            logger.info(f"Trying alternative home directory: {alt_dir}")
            os.makedirs(alt_dir, exist_ok=True)
            
            # Try to set permissions
            try:
                os.chmod(alt_dir, 0o755)  # rwxr-xr-x
            except Exception as e:
                logger.warning(f"Could not set permissions on alternative directory: {e}")
                
            # Update the persist directory
            self.persist_dir = alt_dir
            
            # Create a new storage context
            chroma_dir = os.path.join(alt_dir, "chroma")
            os.makedirs(chroma_dir, exist_ok=True)
            
            try:
                chroma_client = chromadb.PersistentClient(
                    path=chroma_dir
                )
                collection = chroma_client.get_or_create_collection(
                    name="document_collection"
                )
                self.vector_store = ChromaVectorStore(chroma_collection=collection)
                self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
                logger.success("Successfully created alternative storage context")
                return True
            except Exception as e:
                logger.error(f"Failed to create alternative storage context: {e}")
                return False
        
        # Check if index exists
        index_exists = self._check_index_exists()
        
        # Case 1: Force reindex requested
        if self.force_reindex:
            logger.info("Force reindex requested - creating new index")
            if document_paths:
                try:
                    documents = self._load_documents(document_paths)
                    return self._create_index(documents)
                except Exception as e:
                    if "readonly database" in str(e).lower():
                        logger.warning("Readonly database error during force reindex")
                        if try_home_directory_fallback():
                            documents = self._load_documents(document_paths)
                            return self._create_index(documents)
                    else:
                        logger.error(f"Error during force reindex: {e}")
                        raise
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
                if "readonly database" in str(e).lower():
                    logger.warning("Readonly database error when loading existing index")
                    if try_home_directory_fallback() and document_paths:
                        documents = self._load_documents(document_paths)
                        return self._create_index(documents)
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
                if "readonly database" in str(e).lower():
                    logger.warning("Readonly database error when updating existing index")
                    if try_home_directory_fallback():
                        documents = self._load_documents(document_paths)
                        return self._create_index(documents)
                return None
        
        # Case 4: No index exists but documents provided
        if document_paths:
            logger.info("Creating new index from documents")
            try:
                documents = self._load_documents(document_paths)
                return self._create_index(documents)
            except Exception as e:
                if "readonly database" in str(e).lower():
                    logger.warning("Readonly database error when creating new index")
                    if try_home_directory_fallback():
                        documents = self._load_documents(document_paths)
                        return self._create_index(documents)
                logger.error(f"Error creating new index: {e}")
                raise
        
        # Case 5: No index and no documents
        logger.warning("No index exists and no documents provided")
        return None

    def _create_index(self, documents: List[Document]) -> Optional[VectorStoreIndex]:
        """Create vector store index from documents.
        
        Args:
            documents: List of Document objects
            
        Returns:
            Vector store index or None if creation fails
        """
        try:
            if not documents:
                logger.warning("No valid documents provided")
                return None
                
            logger.info("Creating vector index...")
            
            # Ensure storage directory exists with proper permissions
            os.makedirs(self.persist_dir, exist_ok=True)
            
            # Try to set permissions
            try:
                os.chmod(self.persist_dir, 0o755)  # rwxr-xr-x
                
                # Also ensure the chroma directory is writable
                chroma_dir = os.path.join(self.persist_dir, "chroma")
                if os.path.exists(chroma_dir):
                    os.chmod(chroma_dir, 0o755)  # rwxr-xr-x
                    
                    # Check for sqlite file and make it writable
                    sqlite_file = os.path.join(chroma_dir, "chroma.sqlite3")
                    if os.path.exists(sqlite_file):
                        os.chmod(sqlite_file, 0o644)  # rw-r--r--
            except Exception as e:
                logger.warning(f"Could not set permissions: {e}")
            
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
                
                # If it's a readonly database error, try to fix permissions and retry
                if "readonly database" in str(e).lower():
                    logger.warning("Detected readonly database error, attempting to fix permissions...")
                    
                    # Try to fix sqlite file permissions
                    chroma_dir = os.path.join(self.persist_dir, "chroma")
                    sqlite_file = os.path.join(chroma_dir, "chroma.sqlite3")
                    if os.path.exists(sqlite_file):
                        try:
                            os.chmod(sqlite_file, 0o644)  # rw-r--r--
                            logger.info(f"Changed permissions on {sqlite_file}")
                        except Exception as chmod_err:
                            logger.error(f"Failed to change permissions: {chmod_err}")
                    
                    # Try an alternative directory
                    alt_dir = os.path.join(os.path.expanduser("~"), ".temp_rag_storage")
                    logger.info(f"Trying alternative directory: {alt_dir}")
                    os.makedirs(alt_dir, exist_ok=True)
                    
                    # Update the persist directory
                    self.persist_dir = alt_dir
                    
                    # Create a new storage context
                    chroma_client = chromadb.PersistentClient(
                        path=os.path.join(alt_dir, "chroma")
                    )
                    collection = chroma_client.get_or_create_collection(
                        name="document_collection"
                    )
                    
                    self.vector_store = ChromaVectorStore(chroma_collection=collection)
                    self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
                    
                    # Try again with the new directory
                    index = VectorStoreIndex.from_documents(
                        documents,
                        storage_context=self.storage_context,
                        transformations=[self.text_splitter],
                        show_progress=True
                    )
                    
                    self.storage_context.persist(persist_dir=self.persist_dir)
                    logger.success("Successfully persisted index after retry")
            
            return index
            
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            return None

    def _extract_metadata(self, node: Document, doc_id: str = None) -> Dict[str, Any]:
        """Extract metadata from a node.
        
        Args:
            node: The node to extract metadata from
            doc_id: Optional document ID
            
        Returns:
            A dictionary of metadata
        """
        metadata = {}
        
        # Extract file information
        if hasattr(node, "metadata") and node.metadata:
            if "file_path" in node.metadata:
                metadata["source"] = os.path.basename(node.metadata["file_path"])
            
            # Extract page numbers if available
            if "page_label" in node.metadata:
                metadata["page"] = node.metadata["page_label"]
            elif "page" in node.metadata:
                metadata["page"] = str(node.metadata["page"])
                
            # Extract section information if available
            if "section" in node.metadata:
                metadata["section"] = node.metadata["section"]
            
            # Extract title if available
            if "title" in node.metadata:
                metadata["title"] = node.metadata["title"]
                
            # Extract any other useful metadata
            for key in ["chapter", "heading", "subheading"]:
                if key in node.metadata:
                    metadata[key] = node.metadata[key]
        
        return metadata

    def _generate_llm_answer(self, query: str, context: str) -> str:
        """Generate a coherent answer using LLM.
        
        Args:
            query: The query to answer
            context: The context from document sources
            
        Returns:
            A coherent answer generated by the LLM
        """
        try:
            logger.info("Generating answer using LLM")
            
            prompt = f"""\
You are an expert document analyst and information synthesizer. 
You always answer in french unless it's specified by the user

Based on the following sources extracted from documents, provide a comprehensive, 
well-structured answer to this query: "{query}"

Here are the relevant document excerpts:

{context}

Instructions:
1. Synthesize the information from all sources into a coherent, well-structured answer
2. Organize your response with the following sections:
   - Executive Summary (brief overview of the answer)
   - Detailed Analysis (comprehensive explanation with specific details)
   - Key Points (bullet points of the most important information)
3. Focus on directly answering the query with the most relevant information
4. If the sources don't contain enough information to answer the query, acknowledge this limitation
5. Write in a professional, clear, and authoritative style appropriate for analysis
6. Format your response using Markdown for better readability
7. Do not include phrases like "Based on the sources" or "According to the document" - write as if you're providing the information directly
8. Include all relevant details from the sources that help answer the query
9. If there are any ambiguities or potential interpretations, note them clearly
10. Conclude with any recommendations or next steps if appropriate

Your comprehensive answer:
"""
            legal_prompt = f"""\
Vous êtes un expert juridique algérien spécialisé dans l'analyse et l'interprétation du droit algérien.
Sur la base des sources juridiques algériennes suivantes, fournissez une réponse professionnelle, 
complète et bien structurée à cette question: "{query}" 

Voici les extraits pertinents des documents juridiques algériens:

{context}

Instructions pour votre réponse:
1. Synthétisez l'information de toutes les sources en une réponse juridique cohérente et professionnelle
2. Organisez votre réponse avec les sections suivantes:
   - Résumé Exécutif (aperçu concis de la réponse)
   - Analyse Juridique Détaillée (explication complète avec références précises aux articles de loi)
   - Points Clés (points essentiels à retenir)
   - Références Légales (citations précises des articles et textes juridiques pertinents)
3. Concentrez-vous sur une réponse directe à la question avec l'information juridique la plus pertinente
4. Si les sources ne contiennent pas suffisamment d'information pour répondre à la question, reconnaissez cette limitation
5. Écrivez dans un style professionnel, clair et autoritaire approprié pour une analyse juridique algérienne
6. Formatez votre réponse en utilisant Markdown pour une meilleure lisibilité
7. Incluez toutes les références précises aux articles de loi et à la jurisprudence algérienne pertinente
8. Citez correctement les articles du code civil, code de commerce, ou autres textes juridiques algériens
9. Si la question concerne un cas pratique, proposez une application concrète des principes juridiques
10. Concluez avec des recommandations juridiques si approprié

Votre réponse juridique professionnelle:
"""
            
            
            # Generate the answer using LLMTool
            from quantalogic.tools.llm_tool import LLMTool
            
            # Initialize the LLM tool with the same model as specified in the class
            llm_tool = LLMTool(model_name=self.llm_model, event_emitter=self.event_emitter, on_token=self.on_token)
            
            # Execute the tool with system prompt and user query
            system_prompt = "You are an expert in document analysis."
            answer = llm_tool.execute(
                system_prompt=system_prompt,
                prompt=prompt,
                temperature="0.2"
            )
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating LLM answer: {e}")
            return f"Error generating answer: {str(e)}"

    def execute(
        self,
        query: str,
        max_sources: int = 5,
        min_relevance: float = 0.1,
        use_llm: bool = True
    ) -> Dict[str, Any]:
        """Execute the query using SimpleRagTool.
        
        Args:
            query: The query to execute
            max_sources: Maximum number of sources to return
            min_relevance: Minimum relevance score for sources
            use_llm: Whether to use LLM for answer generation
            
        Returns:
            A dictionary with the answer and sources
        """
        try:
            logger.info(f"Executing query: {query}")
            
            # Check if index is initialized
            if self.index is None:
                logger.error("Index is not initialized. Cannot execute query.")
                
                # Try to create a temporary in-memory index if we have documents
                if hasattr(self, 'temp_documents') and self.temp_documents:
                    logger.info("Attempting to create temporary in-memory index...")
                    try:
                        # Create a simple in-memory index
                        from llama_index.core import Settings
                        Settings.embed_model = self.embed_model
                        temp_index = VectorStoreIndex.from_documents(
                            self.temp_documents,
                            transformations=[self.text_splitter]
                        )
                        
                        # Use this temporary index
                        self.index = temp_index
                        logger.success("Created temporary in-memory index")
                    except Exception as e:
                        logger.error(f"Failed to create temporary index: {e}")
                        return {
                            "answer": "Cannot execute query: The document index is not available. Please check if the documents were properly loaded and indexed.",
                            "sources": []
                        }
                else:
                    return {
                        "answer": "Cannot execute query: The document index is not available. Please check if the documents were properly loaded and indexed.",
                        "sources": []
                    }
            
            # Retrieve nodes from the index
            retriever = self.index.as_retriever(
                similarity_top_k=max_sources * 2  # Retrieve more to filter duplicates
            )
            nodes = retriever.retrieve(query)
            
            # Filter nodes by relevance
            nodes = [node for node in nodes if node.score >= min_relevance]
            
            # Remove duplicate content
            unique_nodes = []
            seen_content = set()
            
            for node in nodes:
                # Create a normalized version of the content for comparison
                normalized_content = ' '.join(node.node.text.lower().split())
                
                # Skip if we've seen this content before
                if normalized_content in seen_content:
                    continue
                
                seen_content.add(normalized_content)
                unique_nodes.append(node)
            
            # Limit to max_sources
            unique_nodes = unique_nodes[:max_sources]
            
            # Extract text and metadata from nodes
            sources = []
            context_text = ""
            
            for i, node in enumerate(unique_nodes):
                # Extract metadata
                metadata = self._extract_metadata(node.node)
                
                # Add to sources
                source = {
                    "text": node.node.text,
                    "metadata": metadata,
                    "score": float(node.score) if hasattr(node, "score") else 0.0
                }
                sources.append(source)
                
                # Add to context text for LLM
                source_header = f"SOURCE {i+1}"
                if metadata:
                    source_details = ", ".join([f"{k}: {v}" for k, v in metadata.items()])
                    source_header += f" ({source_details})"
                
                context_text += f"\n\n{source_header}:\n{node.node.text}"
            
            # Generate answer
            if use_llm and sources:
                answer = self._generate_llm_answer(query, context_text)
            else:
                # Simple concatenation of sources
                answer = f"Query: {query}\n\n"
                if sources:
                    answer += "Sources:\n\n"
                    for i, source in enumerate(sources):
                        metadata_str = ", ".join([f"{k}: {v}" for k, v in source["metadata"].items()])
                        answer += f"Source {i+1} ({metadata_str}):\n{source['text']}\n\n"
                else:
                    answer += "No relevant sources found."
            
            return {
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return {
                "answer": f"Error executing query: {str(e)}",
                "sources": []
            }

if __name__ == "__main__":
    # Example usage
    try:
        # Initialize tool
        tool = SimpleRagTool(
            persist_dir="./storage/simple_rag",
            document_paths=[
                "/home/yarab/Téléchargements/CCAP_AC TMA-DMSP_2025_VF1.pdf"
            ],
            chunk_size=1024,
            chunk_overlap=256,
            force_reindex=True,
            model_name="text-embedding-3-large",
            use_temp_dir=False,
            llm_model="gpt-4o-mini"
        )
        
        # Test query
        test_query = "what are the main subject discussed in my doc?"
        
        result = tool.execute(
            query=test_query,
            max_sources=8,
            min_relevance=0.1,
            use_llm=True,
        )
        print(result)
        
    except Exception as e:
        logger.error(f"Failed to initialize or run tool: {e}")
