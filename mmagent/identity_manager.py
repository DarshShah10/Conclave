from qdrant_client import QdrantClient
from qdrant_client.http import models

class IdentityManager:
    def __init__(self, qdrant_client):
        self.client = qdrant_client

    def resolve_identity(self, embedding, collection_name, threshold=0.75):
        """Checks if an embedding matches a known entity using modern Query API."""
        try:
            # query_points is the modern, preferred method for Qdrant 1.7+
            results = self.client.query_points(
                collection_name=collection_name,
                query=embedding,
                limit=1
            ).points

            if results and results[0].score > threshold:
                return results[0].id 
        except Exception as e:
            print(f"Identity Lookup Error: {e}")
        return None

    def register_identity(self, embedding, identity_id, collection_name):
        """Saves a new unique entity to the global database."""
        try:
            self.client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=identity_id,
                        vector=embedding,
                        payload={"type": "global_identity"}
                    )
                ]
            )
        except Exception as e:
            print(f"Identity Registration Error: {e}")