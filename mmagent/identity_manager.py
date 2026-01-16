from qdrant_client import QdrantClient
from qdrant_client.http import models

class IdentityManager:
    def __init__(self, qdrant_client):
        self.client = qdrant_client

    def resolve_identity(self, embedding, collection_name, threshold=0.8):
        """
        Checks if an embedding (Face/Voice) matches a known entity in Qdrant.
        Returns the existing Global ID or a new one.
        """
        search_result = self.client.search(
            collection_name=collection_name,
            query_vector=embedding,
            limit=1
        )

        if search_result and search_result[0].score > threshold:
            return search_result[0].id  # Found existing person
        return None # New person