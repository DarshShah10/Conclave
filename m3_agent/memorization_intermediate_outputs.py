# Copyright (2025) Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

import os
import json
import logging
import argparse
import base64
import cv2
import numpy as np

from mmagent.utils.video_processing import process_video_clip
from mmagent.face_processing import process_faces
from mmagent.voice_processing import process_voices
from mmagent.src.perception_modules import PerceptionModules

# --------------------------------------------------
# Configs & Logger
# --------------------------------------------------
logger = logging.getLogger(__name__)

processing_config = json.load(open("configs/processing_config.json"))
memory_config = json.load(open("configs/memory_config.json"))

FPS = processing_config["fps"]

# Initialize perception modules (OCR + Objects)
perception = PerceptionModules()

# --------------------------------------------------
# Core Segment Processor (WITH EYES)
# --------------------------------------------------
def process_segment(
    video_graph,
    base64_video,
    base64_frames,
    base64_audio,
    clip_id,
    clip_start_time_sec,
    sample
):
    """
    Process ONE clip window:
    - OCR
    - Object Detection
    - Face Processing
    - Voice Processing

    All detections receive precise absolute timestamps (ms).
    """
    save_path = sample["intermediate_outputs"]
    os.makedirs(save_path, exist_ok=True)

    ocr_results = []
    object_results = []

    for i, frame_b64 in enumerate(base64_frames):
        # Decode frame
        img_bytes = base64.b64decode(frame_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            continue

        # Precise absolute timestamp (ms)
        absolute_time_ms = int(
            (clip_start_time_sec * 1000)
            + (i * (1000 / FPS))
        )

        # ---------------- OCR ----------------
        texts = perception.extract_ocr(frame)
        if texts:
            ocr_results.append({
                "ts_ms": absolute_time_ms,
                "data": texts
            })

        # ------------- OBJECTS ---------------
        objects = perception.extract_objects(frame)
        if objects:
            object_results.append({
                "ts_ms": absolute_time_ms,
                "data": objects
            })

    # Save OCR + Object results
    extra_path = os.path.join(
        save_path, f"clip_{clip_id}_extra.json"
    )
    with open(extra_path, "w") as f:
        json.dump(
            {
                "ocr": ocr_results,
                "objects": object_results,
            },
            f,
            indent=2,
        )

    logger.info(
        f"[Clip {clip_id}] OCR={len(ocr_results)} | Objects={len(object_results)}"
    )

    # ---------------- EXISTING PIPELINE ----------------
    process_voices(
        video_graph,
        base64_audio,
        base64_video,
        save_path=os.path.join(save_path, f"clip_{clip_id}_voices.json"),
        preprocessing=["voice"],
    )

    process_faces(
        video_graph,
        base64_frames,
        save_path=os.path.join(save_path, f"clip_{clip_id}_faces.json"),
        preprocessing=["face"],
    )


# --------------------------------------------------
# Sliding Window Video Processor
# --------------------------------------------------
def process_video_with_sliding_window(
    video_path,
    sample,
    window_size=30,
    overlap=5,
):
    """
    Process a video using overlapping sliding windows.

    window_size: seconds per clip
    overlap: seconds of overlap
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    cap.release()

    logger.info(
        f"Video duration: {duration_sec:.2f}s | FPS: {fps}"
    )

    clip_id = 0
    current_start = 0.0

    while current_start < duration_sec:
        end_time = min(current_start + window_size, duration_sec)

        logger.info(
            f"Processing window {clip_id}: "
            f"{current_start:.2f}s → {end_time:.2f}s"
        )

        # Extract clip
        base64_video, base64_frames, base64_audio = process_video_clip(
            video_path,
            start_time=current_start,
            end_time=end_time,
            fps=FPS,
        )

        if base64_frames:
            process_segment(
                video_graph=None,
                base64_video=base64_video,
                base64_frames=base64_frames,
                base64_audio=base64_audio,
                clip_id=clip_id,
                clip_start_time_sec=current_start,
                sample=sample,
            )

        clip_id += 1
        current_start += (window_size - overlap)


# --------------------------------------------------
# Entry Point
# --------------------------------------------------
def streaming_process_video(sample):
    """
    Entry point for dataset-style processing.
    """
    video_path = sample["video_path"]

    process_video_with_sliding_window(
        video_path=video_path,
        sample=sample,
        window_size=30,
        overlap=5,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="data/data.jsonl")
    args = parser.parse_args()

    with open(args.data_file, "r") as f:
        for line in f:
            streaming_process_video(json.loads(line))
