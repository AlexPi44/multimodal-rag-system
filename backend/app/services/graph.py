from neo4j import GraphDatabase
from typing import List, Dict
from app.models.document import Document


class GraphService:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_document_node(self, document: Document):
        with self.driver.session() as session:
            session.run(
                """
                CREATE (d:Document {
                    id: $id,
                    user_id: $user_id,
                    filename: $filename,
                    file_type: $file_type,
                    created_at: $created_at
                })
                """,
                id=document.id,
                user_id=document.user_id,
                filename=document.filename,
                file_type=document.file_type.value,
                created_at=document.created_at.isoformat()
            )

    def create_chunk_relationships(self, document_id: str, chunks: List[str], entities: List[Dict]):
        with self.driver.session() as session:
            for chunk_id, entity_list in zip(chunks, entities):
                for entity in entity_list:
                    session.run(
                        """
                        MATCH (d:Document {id: $doc_id})
                        MERGE (e:Entity {name: $entity_name, type: $entity_type})
                        CREATE (d)-[:CONTAINS_ENTITY]->(e)
                        """,
                        doc_id=document_id,
                        entity_name=entity['name'],
                        entity_type=entity['type']
                    )

    def query_related_documents(self, entity_name: str, user_id: str) -> List[str]:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (d:Document)-[:CONTAINS_ENTITY]->(e:Entity {name: $entity})
                WHERE d.user_id = $user_id
                RETURN DISTINCT d.id as doc_id
                """,
                entity=entity_name,
                user_id=user_id
            )
            return [record['doc_id'] for record in result]
