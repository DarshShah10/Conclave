from neo4j import GraphDatabase
import json
import logging

logger = logging.getLogger(__name__)

class KnowledgeGraph:
    def __init__(self):
        with open("configs/api_config.json") as f:
            config = json.load(f)["neo4j"]
        self.driver = GraphDatabase.driver(
            config["uri"], 
            auth=(config["user"], config["password"])
        )

    def close(self):
        self.driver.close()

    def create_memory_node(self, node_id, content, node_type, clip_id, video_id):
        """Creates a Memory node and links it to a Clip and Video."""
        with self.driver.session() as session:
            query = """
            MERGE (v:Video {id: $video_id})
            MERGE (c:Clip {id: $clip_id, video_id: $video_id})
            MERGE (v)-[:HAS_CLIP]->(c)
            CREATE (m:Memory {
                id: $node_id, 
                content: $content, 
                type: $node_type
            })
            CREATE (c)-[:HAS_MEMORY]->(m)
            """
            session.run(query, 
                node_id=node_id, content=content, 
                node_type=node_type, clip_id=clip_id, video_id=video_id
            )

    def create_entity_node(self, node_id, entity_type, video_id):
        """Creates an Entity (Face/Voice/Object) node."""
        with self.driver.session() as session:
            query = """
            MERGE (e:Entity {id: $node_id, type: $entity_type, video_id: $video_id})
            """
            session.run(query, node_id=node_id, entity_type=entity_type, video_id=video_id)

    def create_relationship(self, source_id, target_id, rel_type="RELATED_TO"):
        """Connects two nodes (e.g., Memory MENTIONS Entity)."""
        with self.driver.session() as session:
            query = f"""
            MATCH (a {{id: $source_id}}), (b {{id: $target_id}})
            MERGE (a)-[:{rel_type}]->(b)
            """
            session.run(query, source_id=source_id, target_id=target_id)

    def get_connected_context(self, memory_id):
        """
        The 'GraphRAG' secret sauce: 
        Find a memory, find the entities in it, and find other memories 
        linked to those same entities.
        """
        with self.driver.session() as session:
            query = """
            MATCH (m:Memory {id: $memory_id})-[:MENTIONS]->(e:Entity)
            MATCH (other:Memory)-[:MENTIONS]->(e)
            WHERE m <> other
            RETURN e.type as entity_type, other.content as related_content
            """
            result = session.run(query, memory_id=memory_id)
            return [dict(record) for record in result]
        
    def create_ocr_node(self, text, clip_id, video_id, timestamp):
        """Creates a node for detected text (signs, plates, etc)."""
        with self.driver.session() as session:
            query = """
            MERGE (t:Text {content: $text, video_id: $video_id})
            WITH t
            MATCH (c:Clip {id: $clip_id, video_id: $video_id})
            MERGE (c)-[:HAS_TEXT {at_ms: $timestamp}]->(t)
            """
            session.run(query, text=text, clip_id=clip_id, video_id=video_id, timestamp=timestamp)

    def create_object_node(self, label, clip_id, video_id, timestamp):
        """Creates a node for detected objects (cars, laptops, etc)."""
        with self.driver.session() as session:
            query = """
            MERGE (o:Object {label: $label, video_id: $video_id})
            WITH o
            MATCH (c:Clip {id: $clip_id, video_id: $video_id})
            MERGE (c)-[:HAS_OBJECT {at_ms: $timestamp}]->(o)
            """
            session.run(query, label=label, clip_id=clip_id, video_id=video_id, timestamp=timestamp)