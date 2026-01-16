import uuid
import json
from qdrant_client import QdrantClient
from qdrant_client.http import models
from mmagent.utils.chat_api import get_embedding_with_retry # For Text
from mmagent.knowledge_graph import KnowledgeGraph # For Neo4j

class UnifiedMemoryEngine:
    def __init__(self, video_id):
        with open("configs/api_config.json") as f:
            conf = json.load(f)
        self.q_client = QdrantClient(url=conf["qdrant"]["url"], api_key=conf["qdrant"]["api_key"])
        self.kg = KnowledgeGraph()
        self.video_id = video_id
        self.text_model = "text-embedding-3-large"

    def upsert_memory(self, collection, vector, payload, node_type):
        point_id = str(uuid.uuid4())
        payload.update({"video_id": self.video_id, "node_id": point_id, "type": node_type})
        
        # 1. Store in Qdrant (Vector Search)
        self.q_client.upsert(
            collection_name=collection,
            points=[models.PointStruct(id=point_id, vector=vector, payload=payload)]
        )
        
        # 2. Store in Neo4j (Relational Logic)
        if node_type in ['episodic', 'semantic']:
            self.kg.create_memory_node(point_id, payload.get('content', ''), node_type, payload['clip_id'], self.video_id)
        else:
            self.kg.create_entity_node(point_id, node_type, self.video_id)
            
        return point_id

    def add_text_memory(self, text, clip_id, text_type='episodic'):
        # Automatic Embedding
        vector, _ = get_embedding_with_retry(self.text_model, text)
        return self.upsert_memory("text_memories", vector, {"content": text, "clip_id": clip_id}, text_type)

    def add_visual_memory(self, frame_vector, clip_id, timestamp_ms, ocr_text=None):
        """Stores a frame embedding (CLIP) for visual search."""
        payload = {"clip_id": clip_id, "ts": timestamp_ms, "ocr": ocr_text}
        return self.upsert_memory("visual_memories", frame_vector, payload, "frame")

    def add_face_memory(self, embedding, clip_id, metadata):
        return self.upsert_memory("face_memories", embedding, {"clip_id": clip_id, **metadata}, "face")

    def add_voice_memory(self, embedding, clip_id, metadata):
        return self.upsert_memory("voice_memories", embedding, {"clip_id": clip_id, **metadata}, "voice")