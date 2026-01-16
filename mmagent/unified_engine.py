import uuid
import json
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models
from mmagent.utils.chat_api import get_embedding_with_retry
from mmagent.knowledge_graph import KnowledgeGraph
from mmagent.identity_manager import IdentityManager

logger = logging.getLogger(__name__)

class UnifiedMemoryEngine:
    def __init__(self, video_id):
        # 1. Load Config
        with open("configs/api_config.json") as f:
            self.config = json.load(f)
        
        # 2. Initialize Qdrant with explicit settings
        self.q_client = QdrantClient(
            url=self.config["qdrant"]["url"], 
            api_key=self.config["qdrant"]["api_key"],
            prefer_grpc=False # Use HTTP for better compatibility on laptops
        )
        
        # 3. Initialize Identity Manager
        self.id_manager = IdentityManager(self.q_client)
        
        # 4. Initialize Neo4j
        self.kg = KnowledgeGraph()
        
        self.video_id = video_id
        self.text_model = "text-embedding-3-large"

    def upsert_memory(self, collection, vector, payload, node_type):
        point_id = payload.get("node_id", str(uuid.uuid4()))
        payload.update({"video_id": self.video_id, "node_id": point_id, "type": node_type})
        
        # Store in Qdrant
        self.q_client.upsert(
            collection_name=collection,
            points=[models.PointStruct(id=point_id, vector=vector, payload=payload)]
        )
        
        # Store in Neo4j
        if node_type in ['episodic', 'semantic']:
            self.kg.create_memory_node(point_id, payload.get('content', ''), node_type, payload.get('clip_id', 0), self.video_id)
        elif node_type in ['face', 'voice', 'object']:
            self.kg.create_entity_node(point_id, node_type, self.video_id)
            
        return point_id

    def add_text_memory(self, text, clip_id, text_type='episodic'):
        vector, _ = get_embedding_with_retry(self.text_model, text)
        return self.upsert_memory("text_memories", vector, {"content": text, "clip_id": clip_id}, text_type)

    def add_visual_memory(self, frame_vector, clip_id, timestamp_ms, ocr_text=None, objects=None):
        payload = {"clip_id": clip_id, "ts": timestamp_ms, "ocr": ocr_text, "objects": objects}
        node_id = self.upsert_memory("visual_memories", frame_vector, payload, "frame")
        if ocr_text:
            try:
                ocr_data = json.loads(ocr_text)
                for item in ocr_data:
                    self.kg.create_ocr_node(item['text'], clip_id, self.video_id, timestamp_ms)
            except: pass
        return node_id

    def add_face_memory(self, embedding, clip_id, metadata):
        return self.upsert_memory("face_memories", embedding, {"clip_id": clip_id, **metadata}, "face")

    def add_voice_memory(self, embedding, clip_id, metadata):
        return self.upsert_memory("voice_memories", embedding, {"clip_id": clip_id, **metadata}, "voice")

    # --- LEGACY WRAPPERS ---
    def add_img_node(self, face_info):
        return self.add_face_memory(face_info["embeddings"][0], 0, {"base64": face_info["contents"][0]})

    def add_voice_node(self, voice_info):
        return self.add_voice_memory(voice_info["embeddings"][0], 0, {"asr": voice_info["contents"][0]})

    def update_node(self, node_id, info):
        pass

    # --- PRO SEARCH (GraphRAG) ---
    def cloud_search(self, query_text, top_k=5):
        # 1. Vector Search (Find the moment)
        query_vector, _ = get_embedding_with_retry(self.text_model, query_text)
        
        try:
            search_results = self.q_client.query_points(
                collection_name="text_memories",
                query=query_vector,
                query_filter=models.Filter(
                    must=[models.FieldCondition(key="video_id", match=models.MatchValue(value=self.video_id))]
                ),
                limit=top_k
            ).points
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

        enriched_knowledge = []
        for res in search_results:
            # 2. Graph Traversal (Find the identity)
            # We ask Neo4j: "What else is linked to the entities in this moment?"
            mem_id = res.id
            # This Cypher query finds the person's name/role linked to the action
            graph_context = self.kg.get_connected_context(mem_id)
            
            enriched_knowledge.append({
                "moment": res.payload.get("content"),
                "clip": res.payload.get("clip_id"),
                "score": res.score,
                "graph_context": graph_context # This contains the "Who is Kelly Power" part
            })
            
        return enriched_knowledge