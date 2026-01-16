import uuid
import json
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models

logger = logging.getLogger(__name__)

class QdrantVideoGraph:
    def __init__(self, video_id):
        # Load Config
        with open("configs/api_config.json") as f:
            config = json.load(f)["qdrant"]
        
        self.client = QdrantClient(url=config["url"], api_key=config["api_key"])
        self.video_id = video_id # Unique ID for the current video session
        
        # We still keep a local edge map for Neo4j migration later
        self.edges = [] 

    def _generate_id(self):
        return str(uuid.uuid4())

    def add_text_node(self, text_data, clip_id, text_type='episodic'):
        """
        text_data: {'contents': [str], 'embeddings': [list]}
        """
        node_id = self._generate_id()
        
        # Prepare Payload (Metadata)
        payload = {
            "video_id": self.video_id,
            "clip_id": clip_id,
            "content": text_data['contents'][0],
            "type": text_type,
            "node_id": node_id
        }

        # Upsert to Qdrant
        self.client.upsert(
            collection_name="text_memories",
            points=[
                models.PointStruct(
                    id=node_id,
                    vector=text_data['embeddings'][0],
                    payload=payload
                )
            ]
        )
        logger.info(f"Added {text_type} node to Qdrant: {node_id}")
        return node_id

    def add_img_node(self, img_data):
        """
        img_data: {'embeddings': [list], 'contents': [base64_str]}
        """
        node_id = self._generate_id()
        
        # We store the embedding in Qdrant, but we should save the 
        # actual image to disk/cloud storage later. For now, payload.
        payload = {
            "video_id": self.video_id,
            "type": "face",
            "node_id": node_id
            # "image_path": ... (Future improvement)
        }

        self.client.upsert(
            collection_name="face_memories",
            points=[
                models.PointStruct(
                    id=node_id,
                    vector=img_data['embeddings'][0],
                    payload=payload
                )
            ]
        )
        return node_id

    def add_edge(self, source_id, target_id, weight=1.0):
        """
        For now, we store edges in a list. 
        Next step: These go to Neo4j.
        """
        edge = {
            "source": source_id,
            "target": target_id,
            "weight": weight,
            "video_id": self.video_id
        }
        self.edges.append(edge)

    def search_text_nodes(self, query_embedding, top_k=5, text_type=None):
        """
        Professional Vector Search with Filtering
        """
        # Create a filter to only search within THIS video
        search_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="video_id",
                    match=models.MatchValue(value=self.video_id)
                )
            ]
        )
        
        # If we want only episodic or only semantic
        if text_type:
            search_filter.must.append(
                models.FieldCondition(
                    key="type",
                    match=models.MatchValue(value=text_type)
                )
            )

        results = self.client.search(
            collection_name="text_memories",
            query_vector=query_embedding,
            query_filter=search_filter,
            limit=top_k,
            with_payload=True
        )

        # Format results to match what the agent expects
        formatted_results = []
        for res in results:
            formatted_results.append({
                "id": res.id,
                "score": res.score,
                "content": res.payload["content"],
                "clip_id": res.payload["clip_id"]
            })
        
        return formatted_results