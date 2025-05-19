import re
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import spacy
from openai import OpenAI
from loguru import logger
from models import LegalDocument, LegalRelation

class DocumentProcessor:
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
        try:
            self.nlp = spacy.load("fr_core_news_sm")
        except OSError:
            logger.info("Downloading French language model...")
            spacy.cli.download("fr_core_news_sm")
            self.nlp = spacy.load("fr_core_news_sm")
            
    def _chunk_text(self, text: str, max_tokens: int = 4000) -> List[str]:
        """Split text into chunks that respect sentence boundaries and token limits."""
        doc = self.nlp(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sent in doc.sents:
            # Rough estimate: 1 token ≈ 4 characters
            sent_length = len(str(sent)) // 4
            
            if current_length + sent_length > max_tokens and current_chunk:
                # Join the current chunk and add it to chunks
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
            
            current_chunk.append(str(sent))
            current_length += sent_length
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get OpenAI embedding for text. If text is too long, split it and average the embeddings."""
        try:
            chunks = self._chunk_text(text)
            if not chunks:
                logger.error("No chunks created from text")
                return None
                
            # Get embeddings for each chunk
            embeddings = []
            total_chunks = len(chunks)
            for i, chunk in enumerate(chunks, 1):
                logger.info(f"Processing chunk {i}/{total_chunks} ({len(chunk) // 4} estimated tokens)")
                try:
                    response = self.client.embeddings.create(
                        model="text-embedding-3-small",
                        input=chunk
                    )
                    embeddings.append(response.data[0].embedding)
                except Exception as e:
                    logger.error(f"Error processing chunk {i}: {e}")
                    continue
            
            # Average the embeddings if we have at least one successful embedding
            if not embeddings:
                logger.error("No embeddings generated")
                return None
                
            avg_embedding = [sum(x) / len(embeddings) for x in zip(*embeddings)]
            logger.info(f"Successfully generated averaged embedding from {len(embeddings)} chunks")
            return avg_embedding
            
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None

    def extract_legal_entities(self, text: str) -> List[Dict]:
        """Extract legal document entities from text."""
        doc = self.nlp(text)
        entities = []
        
        # Regular expressions for legal document patterns
        patterns = {
            'loi': r'Loi\s+(?:n°\s*)?(\d{2,4}[-–]\d{2,4})',
            'decret': r'Décret\s+(?:n°\s*)?(\d{2,4}[-–]\d{2,4})',
            'ordonnance': r'Ordonnance\s+(?:n°\s*)?(\d{2,4}[-–]\d{2,4})',
            'circulaire': r'Circulaire\s+(?:n°\s*)?(\d{2,4}[-–]\d{2,4})',
            'jo': r'Journal[s]?\s+Officiel[s]?\s+(?:n°\s*)?(\d+)'
        }
        
        for doc_type, pattern in patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append({
                    'type': doc_type,
                    'number': match.group(1),
                    'start': match.start(),
                    'end': match.end(),
                    'text': match.group(0)
                })
        
        return entities

    def extract_relations(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract relations between legal documents."""
        relation_patterns = {
            'modifié_par': r'(?:modifié|modifiée)\s+par\s+((?:la\s+)?(?:Loi|Décret|Ordonnance|Circulaire)\s+(?:n°\s*)?(?:\d{2,4}[-–]\d{2,4}))',
            'complété_par': r'(?:complété|complétée)\s+par\s+((?:la\s+)?(?:Loi|Décret|Ordonnance|Circulaire)\s+(?:n°\s*)?(?:\d{2,4}[-–]\d{2,4}))',
            'abrogé_par': r'(?:abrogé|abrogée)\s+par\s+((?:la\s+)?(?:Loi|Décret|Ordonnance|Circulaire)\s+(?:n°\s*)?(?:\d{2,4}[-–]\d{2,4}))',
            'référence': r'(?:référence|en référence)\s+[àa]\s+((?:la\s+)?(?:Loi|Décret|Ordonnance|Circulaire)\s+(?:n°\s*)?(?:\d{2,4}[-–]\d{2,4}))',
            'remplace': r'(?:remplace|remplacé par)\s+((?:la\s+)?(?:Loi|Décret|Ordonnance|Circulaire)\s+(?:n°\s*)?(?:\d{2,4}[-–]\d{2,4}))'
        }
        
        relations = []
        doc = self.nlp(text)
        
        for relation_type, pattern in relation_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                # Find the nearest legal document mention before the relation
                context_before = text[:match.start()]
                source_docs = self.extract_legal_entities(context_before)
                if source_docs:
                    source_doc = source_docs[-1]  # Take the closest one
                    target_doc = self.extract_legal_entities(match.group(1))[0]
                    relations.append((
                        f"{source_doc['type']}_{source_doc['number']}", 
                        f"{target_doc['type']}_{target_doc['number']}", 
                        relation_type
                    ))
        
        return relations

    def process_document(self, text: str, doc_type: str, number: str, title: str) -> Tuple[Optional[LegalDocument], List[LegalRelation]]:
        """Process a legal document and extract all relevant information."""
        # Create document embedding
        embedding = self.get_embedding(text)
        if embedding is None:
            logger.error(f"Failed to create embedding for document: {doc_type}_{number}")
            return None, []
        
        # Create legal document
        doc = LegalDocument(
            id=f"{doc_type}_{number}",
            type=doc_type,
            number=number,
            title=title,
            content=text,
            date_published=datetime.now(),  # This should be extracted from the text
            embedding=embedding
        )
        
        # Extract relations
        relations = []
        for source_id, target_id, relation_type in self.extract_relations(text):
            relation = LegalRelation(
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                date_effective=datetime.now(),  # This should be extracted from the text
                description=f"Relation {relation_type} extracted automatically"
            )
            relations.append(relation)
        
        return doc, relations
