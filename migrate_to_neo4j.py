import pickle
import json
import mmagent.videograph
import sys
sys.modules["videograph"] = mmagent.videograph
from mmagent.knowledge_graph import KnowledgeGraph

def migrate_to_graph(pkl_path, video_id):
    print(f"Migrating {pkl_path} to Neo4j Knowledge Graph...")
    
    with open(pkl_path, "rb") as f:
        old_vg = pickle.load(f)
    
    kg = KnowledgeGraph()
    
    # We need to map the old integer IDs to the same UUIDs we used in Qdrant
    # For this migration, we'll generate them fresh but keep them consistent
    id_map = {}

    # 1. Create Nodes
    for node_id, node in old_vg.nodes.items():
        # Generate a unique ID (In a real pipeline, this comes from Qdrant)
        uid = f"{video_id}_{node_id}" 
        id_map[node_id] = uid
        
        if node.type in ['episodic', 'semantic']:
            kg.create_memory_node(
                node_id=uid,
                content=node.metadata['contents'][0],
                node_type=node.type,
                clip_id=node.metadata['timestamp'],
                video_id=video_id
            )
        elif node.type in ['img', 'voice']:
            kg.create_entity_node(
                node_id=uid,
                entity_type=node.type,
                video_id=video_id
            )

    # 2. Create Relationships (Edges)
    for (n1, n2), weight in old_vg.edges.items():
        if n1 in id_map and n2 in id_map:
            # Determine relationship type
            t1, t2 = old_vg.nodes[n1].type, old_vg.nodes[n2].type
            rel = "MENTIONS" if t1 in ['episodic', 'semantic'] and t2 in ['img', 'voice'] else "RELATED_TO"
            
            kg.create_relationship(id_map[n1], id_map[n2], rel_type=rel)

    print("Neo4j Migration Complete!")
    kg.close()

if __name__ == "__main__":
    migrate_to_graph("data/memory_graphs/test.pkl", "asfa_conference_video")