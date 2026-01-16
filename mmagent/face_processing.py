# Copyright (2025) Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import json
import os
import logging
import gc
import torch

from insightface.app import FaceAnalysis

from mmagent.src.face_extraction import extract_faces
from mmagent.src.face_clustering import cluster_faces
from mmagent.utils.video_processing import process_video_clip

# 🔥 GLOBAL IDENTITY
from mmagent.identity_manager import IdentityManager
from qdrant_client import QdrantClient

os.environ['ONNXRUNTIME_DEVICE_DISCOVERY_DISABLED'] = '1'
os.environ['ORT_LOGGING_LEVEL'] = '3' 


gc.collect()
torch.cuda.empty_cache()

# -------------------- CONFIG --------------------
processing_config = json.load(open("configs/processing_config.json"))

with open("configs/api_config.json") as f:
    q_conf = json.load(f)["qdrant"]

# q_client = QdrantClient(
#     url=q_conf["url"],
#     api_key=q_conf["api_key"]
# )

# id_manager = IdentityManager(q_client)

face_app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

logger = logging.getLogger(__name__)
cluster_size = processing_config["cluster_size"]
# ------------------------------------------------


class Face:
    def __init__(self, frame_id, bounding_box, face_emb, cluster_id, extra_data):
        self.frame_id = frame_id
        self.bounding_box = bounding_box
        self.face_emb = face_emb
        self.cluster_id = cluster_id
        self.extra_data = extra_data


def get_face(frames):
    extracted_faces = extract_faces(face_app, frames)
    return [
        Face(
            frame_id=f["frame_id"],
            bounding_box=f["bounding_box"],
            face_emb=f["face_emb"],
            cluster_id=f["cluster_id"],
            extra_data=f["extra_data"],
        )
        for f in extracted_faces
    ]


def cluster_face(faces):
    faces_json = [
        {
            "frame_id": f.frame_id,
            "bounding_box": f.bounding_box,
            "face_emb": f.face_emb,
            "cluster_id": f.cluster_id,
            "extra_data": f.extra_data,
        }
        for f in faces
    ]

    clustered_faces = cluster_faces(faces_json, 20, 0.5)

    return [
        Face(
            frame_id=f["frame_id"],
            bounding_box=f["bounding_box"],
            face_emb=f["face_emb"],
            cluster_id=f["cluster_id"],
            extra_data=f["extra_data"],
        )
        for f in clustered_faces
    ]


def process_faces(video_graph, base64_frames, save_path, preprocessing=[]):
    batch_size = 1

    def _process_batch(params):
        frames, offset = params
        faces = get_face(frames)
        for face in faces:
            face.frame_id += offset
        return faces

    def get_embeddings(frames, batch_size):
        num_batches = (len(frames) + batch_size - 1) // batch_size
        batched_frames = [
            (frames[i * batch_size : (i + 1) * batch_size], i * batch_size)
            for i in range(num_batches)
        ]

        faces = []
        with ThreadPoolExecutor(max_workers=num_batches) as executor:
            for batch_faces in tqdm(
                executor.map(_process_batch, batched_frames),
                total=num_batches,
            ):
                faces.extend(batch_faces)

        return cluster_face(faces)

    def establish_mapping(faces, key="cluster_id", filter_fn=None):
        mapping = {}
        for face in faces:
            if filter_fn and not filter_fn(face):
                continue
            cid = face[key]
            mapping.setdefault(cid, []).append(face)

        max_faces = processing_config["max_faces_per_character"]
        for cid in mapping:
            mapping[cid] = sorted(
                mapping[cid],
                key=lambda x: (
                    float(x["extra_data"]["face_detection_score"]),
                    float(x["extra_data"]["face_quality_score"]),
                ),
                reverse=True,
            )[:max_faces]
        return mapping

    def filter_score_based(face):
        return (
            float(face["extra_data"]["face_detection_score"])
            > processing_config["face_detection_score_threshold"]
            and float(face["extra_data"]["face_quality_score"])
            > processing_config["face_quality_score_threshold"]
        )

    # 🔥 GLOBAL IDENTITY AWARE GRAPH UPDATE
    def update_videograph(video_graph, tempid2faces):
        id2faces = {}

        for tempid, faces in tempid2faces.items():
            if tempid == -1 or not faces:
                continue

            primary_face = faces[0]
            face_emb = primary_face["face_emb"]
            face_info = {
                "embeddings": [f["face_emb"] for f in faces],
                "contents": [f["extra_data"]["face_base64"] for f in faces],
            }

            # Use the manager attached to the video_graph (the engine)
            global_id = video_graph.id_manager.resolve_identity(
                face_emb, collection_name="face_memories"
            )

            if global_id:
                matched_node = global_id
                video_graph.update_node(matched_node, face_info)
            else:
                matched_node = video_graph.add_img_node(face_info)
                # Register the new identity
                video_graph.id_manager.register_identity(
                    embedding=face_emb,
                    identity_id=matched_node,
                    collection_name="face_memories",
                )

            for f in faces:
                f["matched_node"] = matched_node

            id2faces.setdefault(matched_node, []).extend(faces)

        return id2faces

    # ---------- LOAD OR COMPUTE ----------
    try:
        with open(save_path, "r") as f:
            faces_json = json.load(f)
    except Exception:
        faces = get_embeddings(base64_frames, batch_size)
        faces_json = [
            {
                "frame_id": f.frame_id,
                "bounding_box": f.bounding_box,
                "face_emb": f.face_emb,
                "cluster_id": int(f.cluster_id),
                "extra_data": f.extra_data,
            }
            for f in faces
        ]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(faces_json, f)

    if "face" in preprocessing or not faces_json:
        return {}

    tempid2faces = establish_mapping(
        faces_json, key="cluster_id", filter_fn=filter_score_based
    )

    if not tempid2faces:
        return {}

    return update_videograph(video_graph, tempid2faces)


def main():
    _, frames, _ = process_video_clip(
        "/mnt/hdfs/foundation/longlin.kylin/mmagent/data/video_clips/CZ_2/-OCrS_r5GHc/11.mp4"
    )
    process_faces(
        None, frames, "data/temp/face_detection_results.json", preprocessing=["face"]
    )


if __name__ == "__main__":
    main()
