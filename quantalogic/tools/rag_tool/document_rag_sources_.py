"""Multilingual RAG Tool optimized for French and Arabic using HuggingFace models.

This tool provides enhanced RAG capabilities with:
- Multilingual support (French/Arabic) using specialized embedding models
- Improved query processing with source attribution
- Persistent ChromaDB storage
- Enhanced response formatting
"""

import os
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import asyncio
import shutil
import json
from datetime import datetime

import chromadb
from sentence_transformers import SentenceTransformer
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    Response,
    QueryBundle,
    Settings,
    Document,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.readers.file.docs import PDFReader
from loguru import logger
from quantalogic.tools.tool import Tool, ToolArgument
from quantalogic.tools.rag_tool.ocr_pdf_markdown import PDFToMarkdownConverter
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from huggingface_hub import login

# Configure tool-specific logging
logger.remove()
logger.add(
    sink=lambda msg: print(msg, end=""),
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
)

# Login to Hugging Face
hf_token = os.getenv("HUGGING_FACE_API_KEY")
if hf_token:
    try:
        login(token=hf_token)
        logger.info("Successfully logged in to Hugging Face")
    except Exception as e:
        logger.error(f"Failed to login to Hugging Face: {e}")
else:
    logger.warning("No Hugging Face API token found in environment variables")

@dataclass
class LegalContext:
    """Structured information about a legal document."""
    document_type: Optional[str] = None
    jurisdiction: Optional[str] = None
    court_level: Optional[str] = None
    decision_date: Optional[str] = None
    key_concepts: Optional[List[str]] = None
    temporal_info: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "document_type": self.document_type,
            "jurisdiction": self.jurisdiction,
            "court_level": self.court_level,
            "decision_date": self.decision_date,
            "key_concepts": self.key_concepts,
            "temporal_info": self.temporal_info
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LegalContext':
        """Create from dictionary."""
        return cls(**data)

@dataclass
class LawSource:
    """Structured representation of a law source with enhanced legal context."""
    content: str
    file_name: str
    page_number: str
    reference_number: Optional[str] = None
    score: Optional[float] = None
    legal_context: Optional[LegalContext] = None

@dataclass
class SearchResult:
    """Represents a single search result with combined scores and legal context."""
    content: str
    file_name: str
    page_number: str
    reference_number: Optional[str] = None
    bm25_score: float = 0.0
    embedding_score: float = 0.0
    combined_score: float = 0.0
    metadata: Dict[str, Any] = None
    legal_context: Optional[LegalContext] = None

class LegalTextSplitter(SentenceSplitter):
    """Custom text splitter for legal documents that respects legal document structure."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            paragraph_separator="\n\n"
        )
        
    def split_text(self, text: str) -> List[str]:
        """Split text while preserving legal document structure."""
        # First split by major legal sections
        sections = []
        current_section = []
        
        for line in text.split('\n'):
            line = line.strip()
            # Check for legal section markers
            if any(marker in line.lower() for marker in ['article', 'section', '§', 'chapitre']):
                if current_section:
                    sections.append('\n'.join(current_section))
                current_section = [line]
            else:
                current_section.append(line)
        
        if current_section:
            sections.append('\n'.join(current_section))
        
        # Then apply sentence splitting to each section
        chunks = []
        for section in sections:
            section_chunks = super().split_text(section)
            chunks.extend(section_chunks)
        
        return chunks

class RagToolHf_(Tool):
    """Enhanced RAG tool specialized for law source retrieval."""

    name: str = "rag_tool_hf"
    description: str = (
        "Specialized RAG tool for retrieving and analyzing legal sources "
        "from documents with detailed source attribution."
    )
    arguments: List[ToolArgument] = [
        ToolArgument(
            name="query",
            arg_type="string",
            description="Query to search for specific legal sources",
            required=True,
            example="Find articles related to environmental protection",
        ),
        ToolArgument(
            name="max_sources",
            arg_type="int",
            description="Maximum number of sources to return",
            required=False,
            example="5",
        ),
    ]

    def __init__(
        self,
        name: str = "rag_tool_hf", 
        persist_dir: str = "./storage/multilingual_rag",
        document_paths: Optional[List[str]] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        use_ocr_for_pdfs: bool = False,
        ocr_model: str = "openai/gpt-4o-mini",
        embed_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        force_reindex: bool = False,
        legal_context_model: str = "nlpaueb/legal-bert-base-uncased",
        jurisdiction_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranking_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """Initialize the multilingual RAG tool.
        
        Args:
            force_reindex: If True, forces reindexing even if embeddings exist
        """
        super().__init__()
        self.name = name
        self.persist_dir = os.path.abspath(persist_dir)
        self.use_ocr_for_pdfs = use_ocr_for_pdfs
        self.ocr_model = ocr_model
        self.force_reindex = force_reindex  # Store force_reindex as instance attribute
        
        # Check if we need to reindex
        chroma_persist_dir = os.path.join(self.persist_dir, "chroma")
        embedding_config_path = os.path.join(self.persist_dir, "embedding_config.json")
        needs_reindex = False

        if os.path.exists(embedding_config_path):
            try:
                with open(embedding_config_path, 'r') as f:
                    config = json.load(f)
                if config.get('embed_model') != embed_model:
                    logger.info(f"Embedding model changed from {config.get('embed_model')} to {embed_model}")
                    needs_reindex = True
            except Exception as e:
                logger.warning(f"Failed to read embedding config: {e}")
                needs_reindex = True
        else:
            needs_reindex = True

        # Clean up only if needed
        if (needs_reindex or force_reindex) and os.path.exists(chroma_persist_dir):
            logger.info("Cleaning up existing index due to model change or forced reindex")
            shutil.rmtree(chroma_persist_dir)
            
            # Save new embedding configuration
            os.makedirs(os.path.dirname(embedding_config_path), exist_ok=True)
            with open(embedding_config_path, 'w') as f:
                json.dump({'embed_model': embed_model}, f)

        # Initialize embedding model with error handling
        try:
            self.embed_model = HuggingFaceEmbedding(
                model_name=embed_model,
                embed_batch_size=8
            )
            logger.info(f"Successfully initialized embedding model: {embed_model}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise

        # Initialize legal context models with error handling
        try:
            self.legal_context_model = SentenceTransformer(legal_context_model)
            self.jurisdiction_model = SentenceTransformer(jurisdiction_model)
            self.reranking_model = SentenceTransformer(reranking_model)
            logger.info(f"Successfully initialized legal models")
        except Exception as e:
            logger.error(f"Failed to initialize legal models: {e}")
            raise
        
        # Configure ChromaDB
        try:
            os.makedirs(chroma_persist_dir, exist_ok=True)
            chroma_client = chromadb.PersistentClient(path=chroma_persist_dir)
            collection = chroma_client.create_collection(
                name="multilingual_collection",
                get_or_create=True
            )
            logger.info("Successfully initialized ChromaDB collection")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
        
        self.vector_store = ChromaVectorStore(chroma_collection=collection)
        
        # Configure llama-index settings
        Settings.embed_model = self.embed_model
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap
        Settings.num_output = 1024
        
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        # Initialize text splitter with legal document awareness
        self.text_splitter = LegalTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Initialize or load index
        self.index = self._initialize_index(document_paths)

    async def _process_pdf_with_ocr(self, path: str) -> List[Document]:
        """Process a PDF file using OCR and convert to Documents."""
        try:
            converter = PDFToMarkdownConverter(
                model=self.ocr_model,
                custom_system_prompt=(
                    "Convert the PDF page to clean, well-formatted text. "
                    "Preserve all content including tables, lists, and mathematical notation. "
                    "For images and charts, provide detailed descriptions. "
                    "Maintain the original document structure and hierarchy."
                )
            )
            
            markdown_content = await converter.convert_pdf(path)
            if not markdown_content:
                logger.warning(f"OCR produced no content for {path}")
                return []
                
            # Create a single document with the full content
            doc = Document(
                text=markdown_content,
                metadata={
                    "file_name": os.path.basename(path),
                    "file_path": path,
                    "processing_method": "ocr"
                }
            )
            return [doc]
            
        except Exception as e:
            logger.error(f"Error processing PDF with OCR {path}: {e}")
            return []

    def _load_documents(self, document_paths: List[str]) -> List[Document]:
        """Load documents with special handling for PDFs."""
        all_documents = []
        pdf_reader = PDFReader()
        
        for path in document_paths:
            if not os.path.exists(path):
                logger.warning(f"Document path does not exist: {path}")
                continue
            
            try:
                if path.lower().endswith('.pdf'):
                    if self.use_ocr_for_pdfs:
                        # Use asyncio to run the async OCR function
                        docs = asyncio.run(self._process_pdf_with_ocr(path))
                    else:
                        # Use standard PDF reader
                        docs = pdf_reader.load_data(
                            path,
                            extra_info={
                                "file_name": os.path.basename(path),
                                "file_path": path,
                                "processing_method": "standard"
                            }
                        )
                    
                    if not self.use_ocr_for_pdfs:
                        # Process each page to improve text quality (only for standard PDF reader)
                        processed_docs = []
                        for doc in docs:
                            # Clean up text
                            text = doc.text
                            text = text.replace('\n\n', '[PAGE_BREAK]')
                            text = text.replace('\n', ' ')
                            text = text.replace('[PAGE_BREAK]', '\n\n')
                            text = ' '.join(text.split())
                            
                            processed_doc = Document(
                                text=text,
                                metadata={
                                    **doc.metadata,
                                    "file_name": os.path.basename(path),
                                    "file_path": path,
                                    "page_number": doc.metadata.get("page_number", "unknown"),
                                    "processing_method": "standard"
                                }
                            )
                            processed_docs.append(processed_doc)
                        docs = processed_docs
                else:
                    docs = SimpleDirectoryReader(
                        input_files=[path],
                        filename_as_id=True,
                        file_metadata=lambda x: {"file_name": os.path.basename(x), "file_path": x}
                    ).load_data()
                
                all_documents.extend(docs)
                
                # Log document details
                for doc in docs:
                    logger.debug(f"Document content length: {len(doc.text)} characters")
                    logger.debug(f"Document metadata: {doc.metadata}")
                    preview = doc.text[:200].replace('\n', ' ').strip()
                    logger.debug(f"Content preview: {preview}...")
                
            except Exception as e:
                logger.error(f"Error loading document {path}: {str(e)}")
                continue
                
        return all_documents

    def _initialize_index(self, document_paths: Optional[List[str]]) -> Optional[VectorStoreIndex]:
        """Initialize or load the vector index."""
        logger.info("Initializing index...")
        
        if document_paths:
            return self._create_index(document_paths)
        
        # Try loading existing index
        index_path = os.path.join(self.persist_dir, "docstore.json")
        if os.path.exists(index_path):
            try:
                return load_index_from_storage(storage_context=self.storage_context)
            except Exception as e:
                logger.error(f"Failed to load existing index: {str(e)}")
        else:
            logger.warning("No existing index found and no documents provided")
        
        return None

    def _create_index(self, document_paths: List[str]) -> Optional[VectorStoreIndex]:
        """Create a new index from documents with legal context."""
        try:
            all_documents = self._load_documents(document_paths)

            if not all_documents:
                logger.warning("No valid documents found")
                return None

            # Process documents with legal context
            enhanced_documents = []
            for doc in all_documents:
                # Generate legal context
                legal_context = self._generate_legal_context(doc.text)
                
                # Convert LegalContext to JSON string for ChromaDB storage
                metadata = {
                    "source": doc.metadata.get("source", ""),
                    "legal_context": json.dumps(legal_context.to_dict()) if legal_context else None
                }
                
                enhanced_doc = Document(
                    text=doc.text,
                    metadata=metadata
                )
                enhanced_documents.append(enhanced_doc)

            logger.info(f"Created {len(enhanced_documents)} enhanced documents with legal context")
            
            # Create index with enhanced documents
            index = VectorStoreIndex.from_documents(
                enhanced_documents,
                storage_context=self.storage_context,
                transformations=[self.text_splitter],
                show_progress=True
            )
            
            self.storage_context.persist(persist_dir=self.persist_dir)
            return index

        except Exception as e:
            logger.error(f"Error creating index with legal context: {str(e)}")
            return None

    def _generate_legal_context(self, text: str) -> LegalContext:
        """Generate legal context for a text segment."""
        try:
            # Extract document type and jurisdiction
            doc_type = self._detect_document_type(text)
            jurisdiction = self._detect_jurisdiction(text)
            
            # Extract temporal information
            temporal_info = self._extract_temporal_info(text)
            
            # Extract key legal concepts
            legal_concepts = self._extract_legal_concepts(text)
            
            # Determine court level if applicable
            court_level = self._detect_court_level(text)
            
            return LegalContext(
                document_type=doc_type,
                jurisdiction=jurisdiction,
                court_level=court_level,
                decision_date=temporal_info.get('decision_date'),
                key_concepts=legal_concepts,
                temporal_info=temporal_info
            )
        except Exception as e:
            logger.error(f"Error generating legal context: {e}")
            return None

    def _detect_document_type(self, text: str) -> str:
        """Detect the type of legal document."""
        doc_types = {
            'law': ['loi', 'law', 'قانون'],
            'decree': ['décret', 'decree', 'مرسوم'],
            'regulation': ['règlement', 'regulation', 'تنظيم'],
            'judgment': ['jugement', 'judgment', 'حكم'],
            'contract': ['contrat', 'contract', 'عقد']
        }
        
        text_lower = text.lower()
        for doc_type, keywords in doc_types.items():
            if any(keyword in text_lower for keyword in keywords):
                return doc_type
        return 'other'

    def _detect_jurisdiction(self, text: str) -> str:
        """Detect the jurisdiction of the legal document."""
        # Use jurisdiction model for classification
        embeddings = self.jurisdiction_model.encode(text, convert_to_tensor=True)
        # Add your jurisdiction classification logic here
        return "unknown"

    def _extract_temporal_info(self, text: str) -> Dict[str, Any]:
        """Extract temporal information from the text."""
        import re
        from datetime import datetime
        
        temporal_info = {
            'decision_date': None,
            'effective_date': None,
            'relevant_periods': []
        }
        
        # Date patterns for different formats
        date_patterns = [
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{4})',
            r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})',
            r'(\d{1,2}\s+(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{4})'
        ]
        
        for pattern in date_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                date_str = match.group(1)
                try:
                    # Parse and store the date
                    date = datetime.strptime(date_str, '%Y-%m-%d')
                    temporal_info['decision_date'] = date.isoformat()
                    break
                except ValueError:
                    continue
        
        return temporal_info

    def _extract_legal_concepts(self, text: str) -> List[str]:
        """Extract key legal concepts from the text."""
        # Add your legal concept extraction logic here
        # This could use NER, keyword extraction, or a specialized model
        return []

    def _detect_court_level(self, text: str) -> Optional[str]:
        """Detect the court level from the text."""
        court_levels = {
            'supreme': ['cour suprême', 'supreme court', 'المحكمة العليا'],
            'appeal': ['cour d\'appel', 'court of appeal', 'محكمة الاستئناف'],
            'first_instance': ['tribunal de première instance', 'court of first instance', 'محكمة ابتدائية']
        }
        
        text_lower = text.lower()
        for level, keywords in court_levels.items():
            if any(keyword in text_lower for keyword in keywords):
                return level
        return None

    def _extract_law_reference(self, text: str) -> Optional[str]:
        """Extract law reference numbers from text."""
        import re
        
        # Common patterns for law references
        patterns = [
            r'(?:loi|décret|arrêté)\s+n[°o]?\s*(\d+[-./]\d+)',  # French
            r'(?:قانون|مرسوم|قرار)\s+(?:رقم\s+)?(\d+[-./]\d+)',  # Arabic
            r'(?:law|decree)\s+(?:no\.\s+)?(\d+[-./]\d+)',       # English
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(1)
        return None

    def execute(self, query: str, max_sources: int = 3) -> str:
        """
        Execute a search for legal sources and return a JSON string of law sources.
        
        Args:
            query: Search query for finding relevant law sources
            max_sources: Maximum number of sources to return
            
        Returns:
            JSON string containing an array of law sources with their content and metadata
        """
        try:
            if not self.index:
                raise ValueError("No index available. Please add documents first.")

            logger.info(f"Searching for legal sources with query: {query}")
            
            query_engine = self.index.as_query_engine(
                similarity_top_k=max_sources,
                node_postprocessors=[
                    SimilarityPostprocessor(similarity_cutoff=0.1)
                ],
                response_mode="no_text",
                streaming=False,
                verbose=True
            )
            
            response = query_engine.query(query)
            
            # Process sources
            processed_sources = []
            for node in response.source_nodes:
                if node.score < 0.1:
                    continue
                
                # Extract reference number once to avoid duplicate processing
                ref_number = self._extract_law_reference(node.node.text)
                
                # Parse legal context from JSON string
                legal_context_str = node.metadata.get("legal_context")
                legal_context = None
                if legal_context_str:
                    try:
                        legal_context_dict = json.loads(legal_context_str)
                        legal_context = LegalContext.from_dict(legal_context_dict)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse legal context: {legal_context_str}")

                # Create a dictionary with source information
                source_data = {
                    'content': node.node.text.strip(),
                    'file_name': node.node.metadata.get('file_name', 'Unknown'),
                    'page_number': str(node.node.metadata.get('page_number', 'N/A')),
                    'reference_number': ref_number,
                    'score': float(node.score) if node.score else 0.0,
                    'metadata': {
                        'source_type': 'law_document',
                        'processing_method': node.node.metadata.get('processing_method', 'standard'),
                        'query': query,
                        'timestamp': str(datetime.now().isoformat())
                    },
                    'legal_context': legal_context.to_dict() if legal_context else None
                }
                processed_sources.append(source_data)
            
            # Sort sources by score
            processed_sources.sort(key=lambda x: x['score'], reverse=True)
            
            logger.info(f"Found {len(processed_sources)} relevant law sources for query: {query}")
            return json.dumps(processed_sources, indent=4, ensure_ascii=False)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Source search failed: {error_msg}")
            error_response = {
                'error': error_msg,
                'query': query,
                'timestamp': str(datetime.now().isoformat()),
                'sources': []
            }
            return json.dumps(error_response, indent=4, ensure_ascii=False)

    def format_sources(self, sources: List[LawSource]) -> str:
        """Format a list of LawSource objects into a readable string."""
        if not sources:
            return "No relevant legal sources found in the documents."

        output = ["# Legal Sources Found\n"]
        current_file = None
        
        for source in sources:
            if current_file != source.file_name:
                current_file = source.file_name
                output.append(f"\n## Document: {source.file_name}\n")
            
            # Format source information
            if source.reference_number:
                output.append(f"**Reference Number:** {source.reference_number}\n")
            output.append(f"**Page:** {source.page_number}\n")
            if source.score:
                output.append(f"**Relevance Score:** {round(source.score * 100, 2)}%\n")
            output.append(f"\n{source.content}\n")
            output.append("\n---\n")
        
        return "\n".join(output)

    def add_documents(self, document_paths: List[str]) -> bool:
        """Add new documents to the index."""
        try:
            new_index = self._create_index(document_paths)
            if new_index:
                self.index = new_index
                return True
            return False
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            return False


class RagToolHf(RagToolHf_):
    """Enhanced RAG tool with hybrid BM25 + Embeddings search."""

    def __init__(
        self,
        name: str = "hybrid_rag_tool_hf",
        persist_dir: str = "./storage/hybrid_multilingual_rag",
        document_paths: Optional[List[str]] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        use_ocr_for_pdfs: bool = False,
        ocr_model: str = "openai/gpt-4o-mini",
        embed_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        bm25_weight: float = 0.3,  # Weight for BM25 scores in hybrid ranking
        embedding_weight: float = 0.7,  # Weight for embedding scores in hybrid ranking
        force_reindex: bool = False,
        legal_context_model: str = "nlpaueb/legal-bert-base-uncased",
        jurisdiction_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranking_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """Initialize the hybrid RAG tool with both BM25 and embeddings capabilities.
        
        Args:
            bm25_weight: Weight for BM25 scores in hybrid ranking (0.0-1.0)
            embedding_weight: Weight for embedding scores in hybrid ranking (0.0-1.0)
            force_reindex: If True, forces reindexing even if embeddings exist
        """
        super().__init__(
            name=name,
            persist_dir=persist_dir,
            document_paths=document_paths,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_ocr_for_pdfs=use_ocr_for_pdfs,
            ocr_model=ocr_model,
            embed_model=embed_model,
            force_reindex=force_reindex,
            legal_context_model=legal_context_model,
            jurisdiction_model=jurisdiction_model,
            reranking_model=reranking_model
        )
        
        self.bm25_weight = bm25_weight
        self.embedding_weight = embedding_weight
        
        # Initialize BM25 index and document store
        self.bm25_index = None
        self.document_store = []
        
        # Build BM25 index if we have documents
        if document_paths:
            self._build_hybrid_index(document_paths)

    def _build_hybrid_index(self, document_paths: List[str]):
        """Build BM25 index and optionally rebuild embedding index."""
        # Load documents if needed
        if not self.document_store:
            documents = self._load_documents(document_paths)
            
            # Store documents and their text for BM25
            tokenized_corpus = []
            
            for doc in documents:
                # Process text for BM25
                text = doc.text.lower()
                tokens = text.split()
                
                # Store document info
                self.document_store.append({
                    'text': doc.text,
                    'metadata': doc.metadata,
                    'tokens': tokens
                })
                
                tokenized_corpus.append(tokens)
            
            # Create BM25 index
            self.bm25_index = BM25Okapi(tokenized_corpus)
        
        # Rebuild embedding index if needed
        if not self.index:
            self._create_index(document_paths)

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to range [0, 1] using min-max scaling."""
        if not scores:
            return scores
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(np.array(scores).reshape(-1, 1))
        return normalized.flatten().tolist()

    def execute(self, query: str, max_sources: int = 3) -> str:
        """Execute hybrid search combining BM25 and embedding-based retrieval."""
        try:
            if not self.index or not self.bm25_index:
                raise ValueError("Indices not initialized. Please add documents first.")

            logger.info(f"Executing hybrid search for query: {query}")
            
            # 1. Get embedding-based results
            query_engine = self.index.as_query_engine(
                similarity_top_k=max_sources * 2,  # Get more results for reranking
                node_postprocessors=[
                    SimilarityPostprocessor(similarity_cutoff=0.1)
                ],
                response_mode="no_text",
                streaming=False,
                verbose=True
            )
            
            embedding_response = query_engine.query(query)
            
            # 2. Get BM25 results
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25_index.get_scores(tokenized_query)
            
            # 3. Combine and rank results
            combined_results = []
            seen_texts = set()
            
            # Process embedding results
            for node in embedding_response.source_nodes:
                if node.score < 0.1:
                    continue
                
                text = node.node.text.strip()
                if text in seen_texts:
                    continue
                seen_texts.add(text)
                
                # Find corresponding BM25 score
                doc_idx = next(
                    (i for i, doc in enumerate(self.document_store) 
                     if doc['text'].strip() == text),
                    None
                )
                
                bm25_score = bm25_scores[doc_idx] if doc_idx is not None else 0.0
                
                result = SearchResult(
                    content=text,
                    file_name=node.node.metadata.get('file_name', 'Unknown'),
                    page_number=str(node.node.metadata.get('page_number', 'N/A')),
                    reference_number=self._extract_law_reference(text),
                    bm25_score=bm25_score,
                    embedding_score=float(node.score) if node.score else 0.0,
                    metadata={
                        'source_type': 'law_document',
                        'processing_method': node.node.metadata.get('processing_method', 'standard'),
                        'query': query,
                        'timestamp': str(datetime.now().isoformat())
                    },
                    legal_context=node.node.metadata.get('legal_context')
                )
                combined_results.append(result)
            
            # Normalize scores
            if combined_results:
                bm25_scores = [r.bm25_score for r in combined_results]
                embedding_scores = [r.embedding_score for r in combined_results]
                
                normalized_bm25 = self._normalize_scores(bm25_scores)
                normalized_embedding = self._normalize_scores(embedding_scores)
                
                # Calculate combined scores
                for i, result in enumerate(combined_results):
                    result.bm25_score = normalized_bm25[i]
                    result.embedding_score = normalized_embedding[i]
                    result.combined_score = (
                        self.bm25_weight * result.bm25_score +
                        self.embedding_weight * result.embedding_score
                    )
            
            # Sort by combined score and limit results
            combined_results.sort(key=lambda x: x.combined_score, reverse=True)
            combined_results = combined_results[:max_sources]
            
            # Format results for output
            output_results = []
            for result in combined_results:
                output_results.append({
                    'content': result.content,
                    'file_name': result.file_name,
                    'page_number': result.page_number,
                    'reference_number': result.reference_number,
                    'scores': {
                        'bm25_score': round(result.bm25_score, 4),
                        'embedding_score': round(result.embedding_score, 4),
                        'combined_score': round(result.combined_score, 4)
                    },
                    'metadata': result.metadata,
                    'legal_context': result.legal_context
                })
            
            logger.info(f"Found {len(output_results)} relevant sources using hybrid search")
            return json.dumps(output_results, indent=4, ensure_ascii=False)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Hybrid search failed: {error_msg}")
            error_response = {
                'error': error_msg,
                'query': query,
                'timestamp': str(datetime.now().isoformat()),
                'sources': []
            }
            return json.dumps(error_response, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    # Example usage of hybrid tool
    try:
        if os.path.exists("./storage/hybrid_multilingual_rag"):
            shutil.rmtree("./storage/hybrid_multilingual_rag")
        
        logger.info("Initializing hybrid RAG tool...")
        hybrid_tool = RagToolHf(
            persist_dir="./storage/hybrid_multilingual_rag",
            document_paths=[
                "./docs/test/code_civile.md",
                "./docs/test/code_procedure.md"
            ],
            chunk_size=512,
            chunk_overlap=50,
            use_ocr_for_pdfs=False,
            bm25_weight=0.3,
            embedding_weight=0.7
        )
        
        # Test queries
        test_queries = [
            "Articles du Code Civil Algérien concernant les ouvertures sur propriétés voisines et protection de la vie privée",
            #"Mon voisin a créé des ouvertures (fenêtres) donnant directement sur ma propriété, ce qui porte atteinte à ma vie privée. Je souhaite faire valoir mes droits et le contraindre à fermer ces ouvertures.",
        ]
        
        for query in test_queries:
            logger.info(f"\nExecuting query: {query}")
            try:
                result = hybrid_tool.execute(query, max_sources=2)
                print(json.dumps(json.loads(result), indent=2, ensure_ascii=False))
            except Exception as e:
                logger.error(f"Error executing query '{query}': {e}")
                continue

    except Exception as e:
        logger.error(f"Failed to initialize or run hybrid RAG tool: {e}")
