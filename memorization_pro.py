import json
import glob
import os
import cv2
import base64
import numpy as np
from mmagent.unified_engine import UnifiedMemoryEngine
from mmagent.src.perception_modules import PerceptionModules
from mmagent.identity_manager import IdentityManager
from mmagent.utils.video_processing import process_video_clip
from mmagent.face_processing import process_faces
from mmagent.voice_processing import process_voices
from mmagent.memory_processing import generate_video_context, generate_full_memories

# --- CONFIGURATION ---
VIDEO_ID = "asfa_pro_session_01"
DATA_FILE = "data/data.jsonl"

# Initialize Engines
perception = PerceptionModules()
engine = UnifiedMemoryEngine(video_id=VIDEO_ID)
id_manager = IdentityManager(engine.q_client)

def b64_to_numpy(b64_str):
    img_data = base64.b64decode(b64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def run_pro_pipeline():
    # Load your video data
    with open(DATA_FILE, "r") as f:
        sample = json.loads(f.readline())

    clips = sorted(glob.glob(sample["clip_path"] + "/*.mp4"), 
                   key=lambda x: int(os.path.basename(x).split('.')[0]))

    for clip_path in clips:
        clip_id = int(os.path.basename(clip_path).split('.')[0])
        print(f"\n>>> Processing Clip {clip_id}: {clip_path}")
        
        # 1. Basic Extraction
        v_b64, f_b64, a_b64 = process_video_clip(clip_path)
        
        # 2. Perception & Visual Embedding
        # We sample 1 frame every 2 seconds for visual memory to save space
        for i, frame_b64 in enumerate(f_b64[::10]): 
            frame_np = b64_to_numpy(frame_b64)
            
            # Get CLIP Vector
            visual_vec = perception.get_visual_embedding(frame_np)
            
            # Get OCR/Objects
            ocr = perception.extract_ocr(frame_np)
            objs = perception.extract_objects(frame_np)
            
            # Store in Qdrant/Neo4j
            engine.add_visual_memory(
                visual_vec, 
                clip_id, 
                timestamp_ms=(clip_id * 30000) + (i * 2000),
                ocr_text=json.dumps(ocr),
                objects=json.dumps(objs)
            )

        # 3. Face & Voice with Global Identity Resolution
        id2voices = process_voices(None, a_b64, v_b64, save_path=f"data/temp_v_{clip_id}.json")
        id2faces = process_faces(None, f_b64, save_path=f"data/temp_f_{clip_id}.json")

        # Resolve Faces against Qdrant Global DB
        for f_id, faces in id2faces.items():
            emb = faces[0]["face_emb"]
            gid = id_manager.resolve_identity(emb, "face_memories")
            if not gid:
                gid = engine.add_face_memory(emb, clip_id, {"type": "face"})
            faces[0]["global_id"] = gid # For Neo4j linking

        # 4. Reasoning (Gemini/GPT)
        print("Calling LLM for Episodic/Semantic reasoning...")
        context = generate_video_context(v_b64, f_b64, id2faces, id2voices)
        episodic, semantic = generate_full_memories(context)

        # 5. Unified Storage
        for text in episodic:
            engine.add_text_memory(text, clip_id, "episodic")
        for text in semantic:
            engine.add_text_memory(text, clip_id, "semantic")

    print("\n[SUCCESS] Video fully ingested into Qdrant and Neo4j.")

if __name__ == "__main__":
    run_pro_pipeline()