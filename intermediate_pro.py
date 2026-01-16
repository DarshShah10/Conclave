import os
import json
import cv2
import base64
import numpy as np
import argparse
import logging
from tqdm import tqdm

from mmagent.utils.video_processing import process_video_clip
from mmagent.face_processing import process_faces
from mmagent.voice_processing import process_voices
from mmagent.src.perception_modules import PerceptionModules

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IntermediatePro")

# Load Configs
processing_config = json.load(open("configs/processing_config.json"))
FPS = processing_config["fps"]
perception = PerceptionModules()

def b64_to_numpy(b64_str):
    img_data = base64.b64decode(b64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def process_video_pro(video_path, output_dir, window_size=30, overlap=5):
    """
    Processes video with a sliding window to ensure no events are cut at boundaries.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get video metadata
    cap = cv2.VideoCapture(video_path)
    total_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / total_fps
    cap.release()

    logger.info(f"Processing Video: {video_path} ({duration_sec:.2f}s)")

    clip_id = 0
    current_start = 0.0

    while current_start < duration_sec:
        current_end = min(current_start + window_size, duration_sec)
        logger.info(f"--- Window {clip_id}: {current_start}s to {current_end}s ---")

        # 1. Extract Clip Data (Frames + Audio)
        # Note: We pass start/end to process_video_clip for precise extraction
        v_b64, f_b64, a_b64 = process_video_clip(
            video_path, 
            fps=FPS, 
            start_time=current_start, 
            interval=window_size
        )

        # 2. Run Perception Modules (OCR & Objects)
        ocr_results = []
        obj_results = []
        
        for i, frame_b64 in enumerate(f_b64):
            frame_np = b64_to_numpy(frame_b64)
            # Absolute timestamp in milliseconds
            abs_ts = int((current_start * 1000) + (i * (1000 / FPS)))
            
            # OCR
            texts = perception.extract_ocr(frame_np)
            if texts:
                ocr_results.append({"ts": abs_ts, "data": texts})
            
            # Objects
            objs = perception.extract_objects(frame_np)
            if objs:
                obj_results.append({"ts": abs_ts, "data": objs})

        # 3. Save Master Intermediate JSON for this window
        master_data = {
            "clip_id": clip_id,
            "start_sec": current_start,
            "end_sec": current_end,
            "ocr": ocr_results,
            "objects": obj_results
        }
        
        with open(os.path.join(output_dir, f"clip_{clip_id}_master.json"), "w") as f:
            json.dump(master_data, f, indent=2)

        # 4. Run Legacy Face/Voice Extractors (They save their own JSONs)
        # We pass None for the graph because we only want the JSONs for now
        process_voices(None, a_b64, v_b64, save_path=os.path.join(output_dir, f"clip_{clip_id}_voices.json"))
        process_faces(None, f_b64, save_path=os.path.join(output_dir, f"clip_{clip_id}_faces.json"))

        # Move window forward
        if current_end >= duration_sec:
            break
        current_start += (window_size - overlap)
        clip_id += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to raw mp4")
    parser.add_argument("--out", type=str, required=True, help="Output directory for JSONs")
    args = parser.parse_args()

    process_video_pro(args.video, args.out)