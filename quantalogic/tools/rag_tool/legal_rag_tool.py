"""Algerian Legal RAG Tool using LlamaIndex and ChromaDB for legal document retrieval and analysis.

This tool provides specialized legal RAG capabilities for Algerian law:
- Legal document processing with specialized chunking parameters
- Vector embeddings optimized for legal terminology
- ChromaDB for efficient legal precedent storage
- Professional legal answer format with citations
- Specialized for Algerian civil and commercial law
"""

import os
import shutil
from typing import List, Optional, Dict, Any, Literal
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

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
from quantalogic.tools.tool import Tool, ToolArgument

# Configure tool-specific logging
logger.remove()
logger.add(
    sink=lambda msg: print(msg, end=""),
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
)

@dataclass
class LegalDocumentSource:
    """Structured representation of a legal document source."""
    content: str
    file_name: str
    metadata: Optional[Dict[str, Any]] = None
    score: Optional[float] = None
    legal_citations: Optional[List[str]] = None
    legal_category: Optional[str] = None

class LegalCategory(str, Enum):
    """Categories of Algerian legal documents."""
    CIVIL_CODE = "code_civil"
    COMMERCIAL_CODE = "code_commerce"
    CRIMINAL_CODE = "code_penal"
    PROCEDURE_CODE = "code_procedure"
    FAMILY_CODE = "code_famille"
    CONSTITUTIONAL = "constitutionnel"
    ADMINISTRATIVE = "administratif"
    JURISPRUDENCE = "jurisprudence"
    OTHER = "autre"

class AlgerianLegalRagTool(Tool):
    """Specialized RAG tool for Algerian legal document retrieval and analysis."""

    name: str = "algerian_legal_rag"
    description: str = (
        "Specialized RAG tool for Algerian legal document retrieval and analysis using LlamaIndex and ChromaDB. "
        "Provides professional legal answers with proper citations to Algerian law."
    )
    arguments: List[ToolArgument] = [
        ToolArgument(
            name="query",
            arg_type="string",
            description="Legal query to search for relevant information in Algerian legal documents",
            required=True,
            example="Quelles sont les dispositions du code civil algérien concernant la servitude de vue?",
        ),
        ToolArgument(
            name="max_sources",
            arg_type="int",
            description="Maximum number of legal sources to include in the answer",
            required=False,
            default="8",
        ),
        ToolArgument(
            name="min_relevance",
            arg_type="float",
            description="Minimum relevance score (0-1) for included legal sources",
            required=False,
            default="0.1",
        ),
        ToolArgument(
            name="use_llm",
            arg_type="boolean",
            description="Whether to use LLM to generate a professional legal answer",
            required=False,
            default="True",
        ),
        ToolArgument(
            name="legal_category",
            arg_type="string",
            description="Specific legal category to focus on (civil, commercial, criminal, etc.)",
            required=False,
            default="",
        ),
    ]

    def __init__(
        self,
        name: str = "algerian_legal_rag",
        persist_dir: str = None,
        document_paths: Optional[List[str]] = None,
        chunk_size: int = 1024,
        chunk_overlap: int = 256,
        force_reindex: bool = False,
        model_name: str = "text-embedding-3-large",
        use_temp_dir: bool = True
    ):
        """Initialize the Algerian Legal RAG tool.
        
        Args:
            name: Name of the tool
            persist_dir: Directory to persist the vector store
            document_paths: List of document paths to index
            chunk_size: Size of text chunks for indexing
            chunk_overlap: Overlap between chunks
            force_reindex: Whether to force reindexing
            model_name: OpenAI embedding model name
            use_temp_dir: Whether to use a temporary directory
        """
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
            # If no persist_dir is provided, use a directory in the user's home
            if persist_dir is None:
                home_dir = os.path.expanduser("~")
                persist_dir = os.path.join(home_dir, ".algerian_legal_rag_storage")
                logger.info(f"No persist_dir provided, using home directory: {persist_dir}")
            
            self.persist_dir = os.path.abspath(persist_dir)
            os.makedirs(self.persist_dir, exist_ok=True)
            
            # Try to set permissions
            try:
                os.chmod(self.persist_dir, 0o755)  # rwxr-xr-x
            except Exception as e:
                logger.warning(f"Could not set permissions on persist directory: {e}")
        
        self.force_reindex = force_reindex

        logger.info("Initializing AlgerianLegalRagTool with parameters:")
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
                name="legal_document_collection",
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.success("Successfully initialized ChromaDB collection")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            
            # Try with a fallback directory in user's home
            fallback_dir = os.path.join(os.path.expanduser("~"), ".algerian_legal_rag_storage", "chroma_db")
            logger.info(f"Attempting fallback to user home directory: {fallback_dir}")
            
            # Ensure the fallback directory exists and is writable
            os.makedirs(fallback_dir, exist_ok=True)
            try:
                os.chmod(fallback_dir, 0o755)  # rwxr-xr-x
            except Exception as e:
                logger.warning(f"Could not set permissions on fallback directory: {e}")
            
            # Try with the fallback directory
            chroma_client = chromadb.PersistentClient(path=fallback_dir)
            collection = chroma_client.get_or_create_collection(name="legal_document_collection")
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
        """Extract specialized legal metadata from document.
        
        Args:
            text: Document text
            file_path: Path to the document file
            
        Returns:
            Dictionary of legal metadata
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
        
        # Basic content stats - limit to avoid large metadata
        metadata["char_count"] = len(text)
        metadata["word_count"] = len(text.split())
        
        # Determine legal category from filename or content
        legal_category = self._detect_legal_category(filename, text)
        if legal_category:
            metadata["legal_category"] = legal_category
        
        # Extract legal articles - limit to avoid large metadata
        articles = self._extract_legal_articles(text)
        if articles:
            # Only include first 5 articles to limit metadata size
            metadata["articles"] = articles[:5]
        
        # Try to extract document title and section info from text - with size limits
        lines = text.strip().split('\n')
        if lines:
            # First non-empty line might be a title - limit length
            for line in lines:
                if line.strip():
                    # Limit title length to 100 characters
                    metadata["title"] = line.strip()[:100]
                    break
                    
            # Try to identify legal sections - limit number and size
            sections = []
            for i, line in enumerate(lines):
                if line.strip().lower().startswith(('article', 'section', 'chapitre', 'titre', 'livre')):
                    # Limit each section name to 50 characters
                    sections.append(line.strip()[:50])
                    # Only keep first 10 sections max
                    if len(sections) >= 10:
                        break
            
            if sections:
                metadata["sections"] = sections
        
        # Ensure metadata size is reasonable
        self._limit_metadata_size(metadata)
        
        return metadata
    
    def _limit_metadata_size(self, metadata: Dict[str, Any], max_size: int = 500) -> None:
        """Limit the size of metadata to prevent exceeding chunk size.
        
        Args:
            metadata: Metadata dictionary to limit
            max_size: Maximum size in bytes for any metadata value
        """
        keys_to_check = list(metadata.keys())
        
        for key in keys_to_check:
            value = metadata[key]
            
            # Handle string values
            if isinstance(value, str) and len(value) > max_size:
                metadata[key] = value[:max_size] + "..."
            
            # Handle list values
            elif isinstance(value, list):
                # Limit number of items
                if len(value) > 10:
                    metadata[key] = value[:10]
                
                # Limit size of each item if they're strings
                new_list = []
                for item in metadata[key]:
                    if isinstance(item, str) and len(item) > max_size:
                        new_list.append(item[:max_size] + "...")
                    else:
                        new_list.append(item)
                metadata[key] = new_list
    
    def _detect_legal_category(self, filename: str, text: str) -> Optional[str]:
        """Detect legal category from filename or content.
        
        Args:
            filename: Name of the file
            text: Document text
            
        Returns:
            Legal category or None if not detected
        """
        # Check filename first
        filename_lower = filename.lower()
        if "civil" in filename_lower:
            return LegalCategory.CIVIL_CODE
        elif "commerce" in filename_lower:
            return LegalCategory.COMMERCIAL_CODE
        elif "penal" in filename_lower or "pénal" in filename_lower:
            return LegalCategory.CRIMINAL_CODE
        elif "procedure" in filename_lower or "procédure" in filename_lower:
            return LegalCategory.PROCEDURE_CODE
        elif "famille" in filename_lower or "family" in filename_lower:
            return LegalCategory.FAMILY_CODE
        elif "constitution" in filename_lower:
            return LegalCategory.CONSTITUTIONAL
        elif "admin" in filename_lower:
            return LegalCategory.ADMINISTRATIVE
        elif "jurisprudence" in filename_lower:
            return LegalCategory.JURISPRUDENCE
        
        # Check content if not found in filename
        text_lower = text.lower()
        if "code civil" in text_lower:
            return LegalCategory.CIVIL_CODE
        elif "code de commerce" in text_lower:
            return LegalCategory.COMMERCIAL_CODE
        elif "code pénal" in text_lower or "code penal" in text_lower:
            return LegalCategory.CRIMINAL_CODE
        elif "code de procédure" in text_lower or "code de procedure" in text_lower:
            return LegalCategory.PROCEDURE_CODE
        elif "code de la famille" in text_lower or "code de famille" in text_lower:
            return LegalCategory.FAMILY_CODE
        elif "constitution" in text_lower:
            return LegalCategory.CONSTITUTIONAL
        elif "administratif" in text_lower or "administrative" in text_lower:
            return LegalCategory.ADMINISTRATIVE
        elif "jurisprudence" in text_lower:
            return LegalCategory.JURISPRUDENCE
        
        return LegalCategory.OTHER
    
    def _extract_legal_articles(self, text: str) -> List[str]:
        """Extract legal article references from text.
        
        Args:
            text: Document text
            
        Returns:
            List of article references
        """
        import re
        
        articles = []
        
        # Pattern for article references (e.g., "Article 123", "Art. 456", etc.)
        article_pattern = r'(?:article|art\.?)\s+(\d+(?:\s*(?:et|à|,)\s*\d+)*)'
        article_matches = re.finditer(article_pattern, text.lower())
        
        for match in article_matches:
            articles.append(match.group(0))
        
        return articles

    def _load_documents(self, document_paths: List[str]) -> List[Document]:
        """Load and preprocess legal documents.
        
        Args:
            document_paths: List of paths to legal documents
            
        Returns:
            List of processed legal Document objects
        """
        all_documents = []
        
        for path in document_paths:
            if not os.path.exists(path):
                logger.warning(f"Document path does not exist: {path}")
                continue
                
            try:
                logger.info(f"Loading legal document: {path}")
                docs = SimpleDirectoryReader(
                    input_files=[path],
                    filename_as_id=True
                ).load_data()
                
                logger.info(f"Loaded {len(docs)} legal document(s) from {path}")
                
                # Process each document
                for doc in docs:
                    # Extract specialized legal metadata
                    metadata = self._extract_metadata(doc.text, path)
                    metadata.update(doc.metadata)
                    
                    # Split document into smaller chunks for better legal analysis
                    processed_docs = self._split_legal_document(doc.text, metadata)
                    all_documents.extend(processed_docs)
                    
            except Exception as e:
                logger.error(f"Error loading legal document {path}: {str(e)}")
                continue
        
        # Store documents for potential fallback use
        self.temp_documents = all_documents
                
        return all_documents
    
    def _split_legal_document(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        """Split legal document into smaller chunks based on legal structure.
        
        Args:
            text: Document text
            metadata: Document metadata
            
        Returns:
            List of Document objects
        """
        import re
        
        # Ensure metadata size is reasonable before proceeding
        self._limit_metadata_size(metadata)
        
        # Try to split by articles first
        article_pattern = r'(?:Article|Art\.?)\s+\d+'
        article_splits = re.split(f'({article_pattern})', text, flags=re.IGNORECASE)
        
        # If we have article splits, use them
        if len(article_splits) > 2:  # More than one article found
            documents = []
            current_article = None
            current_text = ""
            
            for i, split in enumerate(article_splits):
                if re.match(article_pattern, split, re.IGNORECASE):
                    # This is an article header
                    if current_article and current_text:
                        # Save the previous article
                        article_metadata = metadata.copy()
                        # Only store the article name, not the full text
                        article_metadata["article"] = current_article[:50] if len(current_article) > 50 else current_article
                        documents.append(Document(
                            text=current_article + " " + current_text.strip(),
                            metadata=article_metadata
                        ))
                    
                    # Start a new article
                    current_article = split
                    current_text = ""
                elif current_article is not None:
                    # Add text to current article
                    current_text += split
            
            # Add the last article
            if current_article and current_text:
                article_metadata = metadata.copy()
                # Only store the article name, not the full text
                article_metadata["article"] = current_article[:50] if len(current_article) > 50 else current_article
                documents.append(Document(
                    text=current_article + " " + current_text.strip(),
                    metadata=article_metadata
                ))
            
            return documents
        
        # If no article splits, use the text splitter
        chunks = self.text_splitter.split_text(text)
        documents = []
        
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = i
            documents.append(Document(
                text=chunk,
                metadata=chunk_metadata
            ))
        
        return documents

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
            alt_dir = os.path.join(home_dir, ".algerian_legal_rag_storage")
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
                    name="legal_document_collection"
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
                        name="legal_document_collection"
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

    def _extract_node_metadata(self, node: Document) -> Dict[str, Any]:
        """Extract metadata from a node for retrieval results.
        
        Args:
            node: The node to extract metadata from
            
        Returns:
            A dictionary of legal metadata
        """
        metadata = {}
        
        # Extract file information
        if hasattr(node, "metadata") and node.metadata:
            # Copy relevant metadata
            for key in ["file_name", "legal_category", "article", "title", "sections"]:
                if key in node.metadata:
                    metadata[key] = node.metadata[key]
            
            # Extract source information
            if "file_path" in node.metadata:
                metadata["source"] = os.path.basename(node.metadata["file_path"])
            
            # Extract article information if available
            if "article" in node.metadata:
                metadata["article"] = node.metadata["article"]
            
            # Extract legal category if available
            if "legal_category" in node.metadata:
                metadata["legal_category"] = node.metadata["legal_category"]
                
            # Extract any other useful legal metadata
            for key in ["chunk_index", "sections"]:
                if key in node.metadata:
                    metadata[key] = node.metadata[key]
        
        return metadata
    
    def _generate_llm_answer(self, query: str, context: str, legal_category: str = "") -> str:
        """Generate a professional legal answer using LLM.
        
        Args:
            query: The legal query to answer
            context: The context from legal document sources
            legal_category: Optional legal category to focus on
            
        Returns:
            A professional legal answer generated by the LLM with proper citations
        """
        try:
            logger.info("Generating professional legal answer using LLM")
            
            # Determine the legal domain focus based on query and category
            domain_focus = ""
            if legal_category:
                domain_focus = f"Focus specifically on Algerian {legal_category.replace('_', ' ')} law. "
            
            # Detect if query is about specific legal concepts
            legal_concepts = []
            for concept in ["servitude", "propriété", "contrat", "responsabilité", "succession", 
                           "mariage", "divorce", "bail", "vente", "hypothèque", "prescription"]:
                if concept.lower() in query.lower():
                    legal_concepts.append(concept)
            
            concepts_guidance = ""
            if legal_concepts:
                concepts_list = ", ".join(legal_concepts)
                concepts_guidance = f"Pay particular attention to the Algerian legal concepts of {concepts_list}. "
            
            prompt = f"""\
Vous êtes un expert juridique algérien spécialisé dans l'analyse et l'interprétation du droit algérien.
Sur la base des sources juridiques algériennes suivantes, fournissez une réponse professionnelle, 
complète et bien structurée à cette question: "{query}"

{domain_focus}{concepts_guidance}

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
            
            # Generate the answer using OpenAI
            from openai import OpenAI
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Vous êtes un expert juridique algérien spécialisé dans l'analyse et l'interprétation du droit algérien, avec une connaissance approfondie du Code Civil, Code de Commerce, et autres textes juridiques algériens."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Lower temperature for more precise legal answers
                max_tokens=3500
            )
            
            # Extract the answer from the response
            answer = response.choices[0].message.content
            
            # Extract legal citations from the answer
            legal_citations = self._extract_legal_citations(answer)
            
            # Format the answer with proper legal formatting
            formatted_answer = self._format_legal_answer(answer, legal_citations)
            
            return formatted_answer
            
        except Exception as e:
            logger.error(f"Error generating legal answer: {e}")
            return f"Erreur lors de la génération de la réponse juridique: {str(e)}"
    
    def _extract_legal_citations(self, text: str) -> List[str]:
        """Extract legal citations from text.
        
        Args:
            text: The text to extract citations from
            
        Returns:
            List of legal citations
        """
        import re
        
        citations = []
        
        # Pattern for article citations (e.g., "article 123", "Art. 456", etc.)
        article_pattern = r'(?:article|art\.?)\s+(\d+(?:\s*(?:et|à|,)\s*\d+)*)'
        article_matches = re.finditer(article_pattern, text.lower())
        
        for match in article_matches:
            citations.append(match.group(0))
        
        # Pattern for code citations (e.g., "Code Civil", "code de commerce", etc.)
        code_pattern = r'(?:code\s+(?:civil|de\s+commerce|de\s+procédure|pénal|de\s+la\s+famille))'
        code_matches = re.finditer(code_pattern, text.lower())
        
        for match in code_matches:
            citations.append(match.group(0))
        
        # Pattern for law citations (e.g., "Loi n° 15-19", etc.)
        law_pattern = r'(?:loi\s+n°\s*\d+[\-\.]\d+)'
        law_matches = re.finditer(law_pattern, text.lower())
        
        for match in law_matches:
            citations.append(match.group(0))
        
        return list(set(citations))  # Remove duplicates
    
    def _format_legal_answer(self, answer: str, citations: List[str]) -> str:
        """Format the legal answer with proper formatting.
        
        Args:
            answer: The raw answer
            citations: List of legal citations
            
        Returns:
            Formatted legal answer
        """
        # If there are citations, add a "Citations Légales" section if not already present
        if citations and "# Citations Légales" not in answer and "## Citations Légales" not in answer:
            answer += "\n\n## Citations Légales\n"
            for citation in citations:
                answer += f"- {citation.capitalize()}\n"
        
        # Ensure proper formatting of legal terms
        answer = answer.replace("article", "Article")
        answer = answer.replace("code civil", "Code Civil")
        answer = answer.replace("code de commerce", "Code de Commerce")
        answer = answer.replace("code pénal", "Code Pénal")
        answer = answer.replace("code de procédure", "Code de Procédure")
        answer = answer.replace("code de la famille", "Code de la Famille")
        
        # Add disclaimer
        if "## Avertissement" not in answer:
            answer += "\n\n## Avertissement\n"
            answer += "Cette analyse juridique est fournie à titre informatif et ne constitue pas un avis juridique formel. "
            answer += "Pour toute question juridique spécifique, veuillez consulter un avocat qualifié en droit algérien."
        
        return answer
    
    def execute(
        self,
        query: str,
        max_sources: int = 5,
        min_relevance: float = 0.1,
        use_llm: bool = True,
        legal_category: str = ""
    ) -> Dict[str, Any]:
        """Execute the query using AlgerianLegalRagTool.
        
        Args:
            query: The query to execute
            max_sources: Maximum number of sources to return
            min_relevance: Minimum relevance score for sources
            use_llm: Whether to use LLM for answer generation
            legal_category: Specific legal category to focus on (civil, commercial, criminal, etc.)
            
        Returns:
            A dictionary with the answer and sources
        """
        try:
            logger.info(f"Executing legal query: {query}")
            
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
                            "answer": "Impossible d'exécuter la requête: L'index de documents juridiques n'est pas disponible. Veuillez vérifier si les documents ont été correctement chargés et indexés.",
                            "sources": []
                        }
                else:
                    return {
                        "answer": "Impossible d'exécuter la requête: L'index de documents juridiques n'est pas disponible. Veuillez vérifier si les documents ont été correctement chargés et indexés.",
                        "sources": []
                    }
            
            # Filter by legal category if provided
            filters = {}
            if legal_category:
                logger.info(f"Filtering by legal category: {legal_category}")
                filters = {"legal_category": legal_category}
            
            # Retrieve nodes from the index
            retriever = self.index.as_retriever(
                similarity_top_k=max_sources * 2,  # Retrieve more to filter duplicates
                filters=filters if filters else None
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
                # Extract metadata using specialized method
                metadata = self._extract_node_metadata(node.node)
                
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
                answer = self._generate_llm_answer(query, context_text, legal_category)
            else:
                # Simple concatenation of sources
                answer = f"Requête: {query}\n\n"
                if sources:
                    answer += "Sources juridiques:\n\n"
                    for i, source in enumerate(sources):
                        metadata_str = ", ".join([f"{k}: {v}" for k, v in source["metadata"].items()])
                        answer += f"Source {i+1} ({metadata_str}):\n{source['text']}\n\n"
                else:
                    answer += "Aucune source juridique pertinente trouvée."
            
            return {
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error executing legal query: {e}")
            return {
                "answer": f"Erreur lors de l'exécution de la requête juridique: {str(e)}",
                "sources": []
            }

if __name__ == "__main__":
    # Example usage
    try:
        # Initialize tool with Algerian legal documents
        tool = AlgerianLegalRagTool(
            persist_dir="./storage/algerian_legal_rag",
            document_paths=[
                "./data/legal/code_civil_algerien.pdf",
                "./data/legal/code_commerce_algerien.pdf",
                "./data/legal/jurisprudence_algerienne.pdf"
            ],
            chunk_size=1024,
            chunk_overlap=256,
            force_reindex=True,
            model_name="text-embedding-3-large",
            use_temp_dir=True  # Use temporary directory for testing
        )
        
        # Example legal queries
        legal_queries = [
            "Quelles sont les dispositions du code civil algérien concernant la servitude de vue?",
            "Comment le droit algérien traite-t-il les conflits de voisinage liés aux fenêtres donnant sur une propriété privée?",
            "Quels sont les recours légaux en Algérie pour contraindre un voisin à fermer des ouvertures (fenêtres) donnant sur ma propriété?"
        ]
        
        # Execute a sample query
        result = tool.execute(
            query=legal_queries[2],  # Third query about legal remedies
            max_sources=8,
            min_relevance=0.1,
            use_llm=True,
            legal_category=LegalCategory.CIVIL_CODE  # Focus on civil code
        )
        
        # Print the result
        print("\n" + "="*80)
        print("REQUÊTE JURIDIQUE:")
        print(legal_queries[2])
        print("\n" + "="*80)
        print("RÉPONSE JURIDIQUE:")
        print(result["answer"])
        print("\n" + "="*80)
        print(f"NOMBRE DE SOURCES: {len(result['sources'])}")
        
    except Exception as e:
        logger.error(f"Failed to initialize or run tool: {e}")
        import sys
        sys.exit(1)
