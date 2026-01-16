import json
import glob
import os
import cv2
import base64
import re
import numpy as np
from mmagent.unified_engine import UnifiedMemoryEngine
from mmagent.src.perception_modules import PerceptionModules
from mmagent.utils.video_processing import process_video_clip
from mmagent.face_processing import process_faces
from mmagent.voice_processing import process_voices
from mmagent.memory_processing import generate_video_context, generate_full_memories

VIDEO_ID = "asfa_pro_session_01"
engine = UnifiedMemoryEngine(video_id=VIDEO_ID)
perception = PerceptionModules()

def b64_to_numpy(b64_str):
    img_data = base64.b64decode(b64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def parse_entities_from_text(text):
    # Finds UUIDs in format <face_UUID> or <voice_UUID>
    return re.findall(r'<(face|voice|character)_([a-z0-9\-]+)>', text)

def run_pro_pipeline():
    with open("data/data.jsonl", "r") as f:
        sample = json.loads(f.readline())

    clips = sorted(glob.glob(sample["clip_path"] + "/*.mp4"), key=lambda x: int(os.path.basename(x).split('.')[0]))

    for clip_path in clips:
        clip_id = int(os.path.basename(clip_path).split('.')[0])
        print(f"\n>>> Processing Clip {clip_id}")
        
        v_b64, f_b64, a_b64 = process_video_clip(clip_path)
        
        # 1. Extract Faces/Voices (This now uses Global ID Resolution internally)
        id2voices = process_voices(engine, a_b64, v_b64, save_path=f"data/temp_v_{clip_id}.json")
        id2faces = process_faces(engine, f_b64, save_path=f"data/temp_f_{clip_id}.json")

        # 2. Visual Sampling (OCR/Objects/CLIP)
        for i, frame_b64 in enumerate(f_b64[::15]): # Sample every 3 seconds
            frame_np = b64_to_numpy(frame_b64)
            ocr = perception.extract_ocr(frame_np)
            objs = perception.extract_objects(frame_np)
            vec = perception.get_visual_embedding(frame_np)
            
            engine.add_visual_memory(vec, clip_id, (clip_id*30000)+(i*3000), json.dumps(ocr), json.dumps(objs))

        # 3. Reasoning with UUIDs
        # We pass the id2faces/id2voices which now contain Global UUIDs
        context = generate_video_context(v_b64, f_b64, id2faces, id2voices)
        episodic, semantic = generate_full_memories(context)

        # 4. Unified Storage with Neo4j Linking
        for text in episodic + semantic:
            mem_type = "episodic" if text in episodic else "semantic"
            # Save to Qdrant & Neo4j
            mem_uuid = engine.add_text_memory(text, clip_id, mem_type)
            
            # 🔥 THE FIX: Link this memory to the specific people mentioned
            mentions = parse_entities_from_text(text)
            for ent_type, ent_uuid in mentions:
                print(f"   [Link] Memory {mem_uuid[:8]} -> Entity {ent_uuid[:8]}")
                engine.link_memory_to_entity(mem_uuid, ent_uuid, "MENTIONS")

    print("\n[SUCCESS] Brain Rebuilt with Deep Linking.")

if __name__ == "__main__":
    run_pro_pipeline()