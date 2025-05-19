from typing import List, Optional
from neo4j import GraphDatabase
from loguru import logger
from models import LegalDocument, LegalRelation, JournalOfficiel

class Neo4jManager:
    def __init__(self, uri: str, username: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self._setup_constraints()

    def _setup_constraints(self):
        with self.driver.session() as session:
            # Create constraints for unique IDs
            constraints = [
                "CREATE CONSTRAINT legal_doc_id IF NOT EXISTS FOR (d:LegalDocument) REQUIRE d.id IS UNIQUE",
                "CREATE CONSTRAINT jo_number IF NOT EXISTS FOR (j:JournalOfficiel) REQUIRE j.number IS UNIQUE"
            ]
            for constraint in constraints:
                session.run(constraint)

    def create_legal_document(self, doc: LegalDocument):
        with self.driver.session() as session:
            query = """
            CREATE (d:LegalDocument {
                id: $id,
                type: $type,
                number: $number,
                title: $title,
                content: $content,
                date_published: datetime($date_published),
                embedding: $embedding
            })
            """
            session.run(query, dict(doc))

    def create_relation(self, relation: LegalRelation):
        with self.driver.session() as session:
            query = """
            MATCH (source:LegalDocument {id: $source_id})
            MATCH (target:LegalDocument {id: $target_id})
            CREATE (source)-[r:LEGAL_RELATION {
                type: $relation_type,
                date_effective: datetime($date_effective),
                description: $description
            }]->(target)
            """
            session.run(query, dict(relation))

    def create_journal_officiel(self, jo: JournalOfficiel):
        with self.driver.session() as session:
            query = """
            CREATE (j:JournalOfficiel {
                number: $number,
                date_published: datetime($date_published),
                content: $content,
                embedding: $embedding
            })
            """
            session.run(query, dict(jo))

    def link_doc_to_jo(self, doc_id: str, jo_number: str):
        with self.driver.session() as session:
            query = """
            MATCH (d:LegalDocument {id: $doc_id})
            MATCH (j:JournalOfficiel {number: $jo_number})
            CREATE (d)-[r:PUBLISHED_IN]->(j)
            """
            session.run(query, {"doc_id": doc_id, "jo_number": jo_number})

    def find_similar_documents(self, embedding: List[float], limit: int = 5):
        with self.driver.session() as session:
            query = """
            CALL db.index.vector.queryNodes('legal_docs_embedding', $limit, $embedding)
            YIELD node, score
            RETURN node.id, node.title, node.type, score
            """
            result = session.run(query, {"embedding": embedding, "limit": limit})
            return [dict(record) for record in result]

    def get_document_relations(self, doc_id: str):
        with self.driver.session() as session:
            query = """
            MATCH (d:LegalDocument {id: $doc_id})-[r:LEGAL_RELATION]-(related)
            RETURN type(r) as relation_type, 
                   r.date_effective as date_effective,
                   r.description as description,
                   related.id as related_doc_id,
                   related.title as related_doc_title
            """
            result = session.run(query, {"doc_id": doc_id})
            return [dict(record) for record in result]

    def close(self):
        self.driver.close()
