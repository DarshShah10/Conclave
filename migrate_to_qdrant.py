import pickle
import mmagent.videograph
import sys
sys.modules["videograph"] = mmagent.videograph
from mmagent.videograph_qdrant import QdrantVideoGraph
import json


def migrate(pkl_path, video_id):
    print(f"Migrating {pkl_path} to Qdrant Cloud...")
    
    with open(pkl_path, "rb") as f:
        old_vg = pickle.load(f)

    # Initialize our new Cloud Graph
    new_vg = QdrantVideoGraph(video_id=video_id)
    
    # Map old integer IDs to new UUIDs
    id_map = {}

    # 1. Migrate Text Nodes
    for node_id, node in old_vg.nodes.items():
        if node.type in ['episodic', 'semantic']:
            text_data = {
                'contents': node.metadata['contents'],
                'embeddings': node.embeddings
            }
            new_id = new_vg.add_text_node(text_data, node.metadata['timestamp'], node.type)
            id_map[node_id] = new_id

    # 2. Migrate Edges (Relationships)
    for (n1, n2), weight in old_vg.edges.items():
        if n1 in id_map and n2 in id_map:
            new_vg.add_edge(id_map[n1], id_map[n2], weight)

    # Save the edges to a temp file for the Neo4j step
    with open(f"data/edges_{video_id}.json", "w") as f:
        json.dump(new_vg.edges, f)

    print(f"Migration complete! {len(id_map)} nodes moved to Qdrant.")

if __name__ == "__main__":
    migrate("data/memory_graphs/test.pkl", "asfa_conference_video")