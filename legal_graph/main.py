import os
from dotenv import load_dotenv
from loguru import logger
from db_manager import Neo4jManager
from document_processor import DocumentProcessor
from file_processor import FileProcessor

# Load environment variables
load_dotenv()

def process_documents(docs_path: str):
    """Process all documents in the specified directory."""
    # Initialize components
    neo4j_manager = Neo4jManager(
        uri="bolt://localhost:7687", #os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        username="neo4j", #os.getenv("NEO4J_USER", "neo4j"),
        password="testLegalPlace" # os.getenv("NEO4J_PASSWORD")
    )
    
    doc_processor = DocumentProcessor(
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    file_processor = FileProcessor(doc_processor)
    
    try:
        # Process all files in directory
        results = file_processor.process_directory(docs_path)
        
        for result in results:
            if not result:
                continue
                
            doc = result['document']
            relations = result['relations']
            metadata = result['metadata']
            
            logger.info(f"Processing document: {metadata['file_path']}")
            
            # Skip if document processing failed
            if doc is None:
                logger.warning(f"Failed to process document: {metadata['file_path']}")
                continue
            
            try:
                # Store document in Neo4j
                neo4j_manager.create_legal_document(doc)
                logger.info(f"Created document node: {doc.id}")
                
                # Store relations
                for relation in relations:
                    try:
                        neo4j_manager.create_relation(relation)
                        logger.info(f"Created relation: {relation.source_id} -{relation.relation_type}-> {relation.target_id}")
                    except Exception as e:
                        logger.error(f"Error creating relation: {e}")
                
                logger.info(f"Successfully processed {metadata['file_path']}")
                
                # Find similar documents
                similar_docs = neo4j_manager.find_similar_documents(doc.embedding)
                if similar_docs:
                    logger.info(f"Similar documents for {doc.id}:")
                    for similar in similar_docs:
                        logger.info(f"  - {similar['node.title']} (score: {similar['score']:.3f})")
                
            except Exception as e:
                logger.error(f"Error processing document {metadata['file_path']}: {e}")
            
    except Exception as e:
        logger.error(f"Error in main processing: {e}")
    
    finally:
        neo4j_manager.close()

if __name__ == "__main__":
    docs_path = "/home/yarab/Bureau/trash_agents_tests/f1/docs/folder_test"
    process_documents(docs_path)
