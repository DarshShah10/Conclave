import json
from qdrant_client import QdrantClient
from qdrant_client.http import models

# 1. Load your credentials (Add these to your configs/api_config.json first!)
# Or just hardcode them here for the first test:
QDRANT_URL = "https://03cac66b-a274-4c83-981f-a5d3fcffe87a.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.NJ8pks5P19sKY08XkrXIB7KDhJvhCp3PavYHq_Yo8F4"

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def create_collections():
    # Collection for Text (Episodic/Semantic)
    # OpenAI text-embedding-3-large is 3072 dimensions
    client.recreate_collection(
        collection_name="text_memories",
        vectors_config=models.VectorParams(size=3072, distance=models.Distance.COSINE),
    )

    # Collection for Faces
    # InsightFace buffalo_l is 512 dimensions
    client.recreate_collection(
        collection_name="face_memories",
        vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE),
    )

    # Collection for Voices
    # SpeakerLab ERes2NetV2 is 192 dimensions
    client.recreate_collection(
        collection_name="voice_memories",
        vectors_config=models.VectorParams(size=192, distance=models.Distance.COSINE),
    )
    
    client.recreate_collection(
        collection_name="visual_memories",
        vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE), # CLIP size
    )

    print("Successfully created text_memories, face_memories, and voice_memories collections!")

if __name__ == "__main__":
    create_collections()