from mmagent.unified_engine import UnifiedMemoryEngine
from mmagent.src.perception_modules import PerceptionModules
from mmagent.memory_processing import generate_video_context, generate_full_memories
from mmagent.identity_manager import IdentityManager

# Initialize Pro Modules
perception = PerceptionModules()
engine = UnifiedMemoryEngine(video_id="my_pro_video_01")
id_manager = IdentityManager(engine.q_client)

def process_pro_pipeline(sample):
    clips = sorted(glob.glob(sample["clip_path"] + "/*.mp4"))
    
    for clip_path in clips:
        clip_id = int(clip_path.split("/")[-1].split(".")[0])
        v_b64, f_b64, a_b64 = process_video_clip(clip_path)
        
        # 1. Perception: Faces, Voices, OCR, Objects
        id2voices = process_voices(None, a_b64, v_b64, save_path=...)
        id2faces = process_faces(None, f_b64, save_path=...)
        
        # 2. Global Identity Resolution & Storage
        for face_id, faces in id2faces.items():
            emb = faces[0]["face_emb"]
            # Check if we've seen this person in Qdrant before
            global_id = id_manager.resolve_identity(emb, "face_memories")
            if not global_id:
                global_id = engine.add_face_memory(emb, clip_id, {"quality": faces[0]["extra_data"]["face_quality_score"]})
            faces[0]["global_id"] = global_id # Link for Neo4j

        # 3. Visual/OCR Embedding (New Modality)
        for i, frame_b64 in enumerate(f_b64):
            # Extract OCR/Objects
            frame_np = b64_to_numpy(frame_b64)
            ocr_data = perception.extract_ocr(frame_np)
            
            # Get Visual Embedding (CLIP)
            # Note: You can use a local CLIP model here to save API costs
            visual_vector = perception.get_visual_embedding(frame_np) 
            
            engine.add_visual_memory(visual_vector, clip_id, timestamp_ms=..., ocr_text=ocr_data)

        # 4. Reasoning: Generate Captions via Gemini/GPT
        context = generate_video_context(v_b64, f_b64, id2faces, id2voices)
        episodic, semantic = generate_full_memories(context)

        # 5. Store Text Memories (Auto-Embeds via OpenAI)
        for text in episodic:
            mem_id = engine.add_text_memory(text, clip_id, "episodic")
            # Link Memory to the Faces/Voices mentioned in it in Neo4j
            entities = parse_entities(text) 
            for ent in entities:
                engine.kg.create_relationship(mem_id, ent["global_id"], "MENTIONS")

        for text in semantic:
            engine.add_text_memory(text, clip_id, "semantic")

    print("Pro Pipeline Complete. Data is live in Qdrant and Neo4j.")