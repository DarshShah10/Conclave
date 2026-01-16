import json
import sys
from qdrant_client import QdrantClient
from qdrant_client.http import models

print("--- Qdrant Index Fixer Starting ---")

# 1. Load Config
try:
    with open("configs/api_config.json") as f:
        config = json.load(f)["qdrant"]
    
    # Clean URL: Remove trailing slashes or ports if they exist
    url = config["url"].replace(":6333", "").strip("/")
    print(f"[*] Target URL: {url}")
    
    client = QdrantClient(url=url, api_key=config["api_key"], timeout=30)
    print("[*] Connection initialized.")
except Exception as e:
    print(f"[!] Setup Error: {e}")
    sys.exit(1)

collections = ["text_memories", "face_memories", "voice_memories", "visual_memories"]

# 2. Create Indexes
for col in collections:
    print(f"\n[*] Processing collection: {col}")
    try:
        # Check if collection exists first
        client.get_collection(col)
        
        print(f"    > Creating 'video_id' index...")
        client.create_payload_index(
            collection_name=col,
            field_name="video_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        
        print(f"    > Creating 'type' index...")
        client.create_payload_index(
            collection_name=col,
            field_name="type",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        print(f"    [OK] Indexes verified for {col}")
    except Exception as e:
        print(f"    [!] Error with {col}: {e}")

print("\n--- All Tasks Finished ---")