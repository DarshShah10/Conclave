# Copyright (2025) Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

import base64
import struct
import json
import os
import logging
import io
import torch
import torchaudio
from io import BytesIO
from pydub import AudioSegment

from speakerlab.process.processor import FBank
from speakerlab.utils.builder import dynamic_import

from .prompts import prompt_audio_segmentation
from .utils.chat_api import generate_messages, get_response
from .utils.general import validate_and_fix_json, normalize_embedding

# 🔥 GLOBAL IDENTITY
from mmagent.identity_manager import IdentityManager
from qdrant_client import QdrantClient

# ---------------- CONFIG ----------------
processing_config = json.load(open("configs/processing_config.json"))
MAX_RETRIES = processing_config["max_retries"]

with open("configs/api_config.json") as f:
    q_conf = json.load(f)["qdrant"]

q_client = QdrantClient(
    url=q_conf["url"],
    api_key=q_conf["api_key"]
)
id_manager = IdentityManager(q_client)
# ---------------------------------------


# ---------------- MODEL -----------------
pretrained_state = torch.load(
    "models/pretrained_eres2netv2.ckpt", map_location="cpu"
)

model_cfg = {
    "obj": "speakerlab.models.eres2net.ERes2NetV2.ERes2NetV2",
    "args": {
        "feat_dim": 80,
        "embedding_size": 192,
    },
}

embedding_model = dynamic_import(model_cfg["obj"])(**model_cfg["args"])
embedding_model.load_state_dict(pretrained_state)
embedding_model.to(torch.device("cuda"))
embedding_model.eval()

feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)
# ---------------------------------------


def get_embedding(wav):

    def load_wav(wav_file, obj_fs=16000):
        wav, fs = torchaudio.load(wav_file)
        if fs != obj_fs:
            wav, _ = torchaudio.sox_effects.apply_effects_tensor(
                wav, fs, effects=[["rate", str(obj_fs)]]
            )
        if wav.shape[0] > 1:
            wav = wav[0:1, :]
        return wav

    wav = load_wav(wav)
    feat = feature_extractor(wav).unsqueeze(0).to(torch.device("cuda"))

    with torch.no_grad():
        emb = embedding_model(feat).squeeze(0).cpu().numpy()

    return emb


@torch.no_grad()
def generate(wav):
    wav = base64.b64decode(wav)
    wav_file = BytesIO(wav)
    return get_embedding(wav_file)


@torch.no_grad()
def get_audio_embeddings(audio_segments):
    res = []
    for wav in audio_segments:
        emb = generate(wav.decode("utf-8"))
        res.append(struct.pack("f" * len(emb), *emb))
    return res


logger = logging.getLogger(__name__)


def process_voices(video_graph, base64_audio, base64_video, save_path, preprocessing=[]):

    def diarize_audio(base64_video, filter_fn):
        inputs = [
            {"type": "video_base64/mp4", "content": base64_video.decode("utf-8")},
            {"type": "text", "content": prompt_audio_segmentation},
        ]
        messages = generate_messages(inputs)

        asrs = None
        for _ in range(MAX_RETRIES):
            response, _ = get_response(
                "gemini-2.5-flash-lite", messages, timeout=30
            )
            asrs = validate_and_fix_json(response)
            if asrs:
                break
        if not asrs:
            raise RuntimeError("Audio diarization failed")

        for a in asrs:
            sm, ss = map(int, a["start_time"].split(":"))
            em, es = map(int, a["end_time"].split(":"))
            a["duration"] = (em * 60 + es) - (sm * 60 + ss)

        return [a for a in asrs if filter_fn(a)]

    def get_audio_segments(base64_audio, dialogs):
        audio = AudioSegment.from_wav(io.BytesIO(base64.b64decode(base64_audio)))
        segments = []

        for start, end in dialogs:
            sm, ss = map(int, start.split(":"))
            em, es = map(int, end.split(":"))
            s_ms = (sm * 60 + ss) * 1000
            e_ms = (em * 60 + es) * 1000

            if s_ms >= e_ms or e_ms > len(audio):
                segments.append(None)
                continue

            buf = io.BytesIO()
            audio[s_ms:e_ms].export(buf, format="wav")
            segments.append(base64.b64encode(buf.getvalue()))

        return segments

    def filter_duration(audio):
        return audio["duration"] >= processing_config["min_duration_for_audio"]

    def get_embeddings(audios):
        segments = [a["audio_segment"] for a in audios]
        embs = get_audio_embeddings(segments)
        for a, e in zip(audios, embs):
            a["embedding"] = normalize_embedding(e)
        return audios

    # 🔥 GLOBAL SPEAKER RESOLUTION
    def update_videograph(video_graph, audios):
        id2audios = {}

        for audio in audios:
            emb = audio["embedding"]

            audio_info = {
                "embeddings": [emb],
                "contents": [audio["asr"]],
            }

            global_id = id_manager.resolve_identity(
                emb, collection_name="voice_memories"
            )

            if global_id:
                matched_node = global_id
                video_graph.update_node(matched_node, audio_info)
            else:
                matched_node = video_graph.add_voice_node(audio_info)
                id_manager.register_identity(
                    embedding=emb,
                    identity_id=matched_node,
                    collection_name="voice_memories",
                )

            audio["matched_node"] = matched_node
            id2audios.setdefault(matched_node, []).append(audio)

        return id2audios

    # --------- LOAD OR PROCESS ----------
    if not base64_audio:
        return {}

    try:
        with open(save_path, "r") as f:
            audios = json.load(f)
        for a in audios:
            a["audio_segment"] = a["audio_segment"].encode("utf-8")
    except Exception:
        asrs = diarize_audio(base64_video, filter_duration)
        dialogs = [(a["start_time"], a["end_time"]) for a in asrs]
        segments = get_audio_segments(base64_audio, dialogs)

        audios = []
        for a, seg in zip(asrs, segments):
            if seg:
                a["audio_segment"] = seg
                audios.append(a)

        if audios:
            audios = get_embeddings(audios)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            for a in audios:
                a["audio_segment"] = a["audio_segment"].decode("utf-8")
            json.dump(audios, f)
            for a in audios:
                a["audio_segment"] = a["audio_segment"].encode("utf-8")

    if "voice" in preprocessing or not audios:
        return {}

    return update_videograph(video_graph, audios)


if __name__ == "__main__":
    pass
