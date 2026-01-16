import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer
from PIL import Image
import torch

class PerceptionModules:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Use a slightly better OCR model configuration
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        # YOLOv10n for speed
        self.object_model = YOLO('yolov10n.pt') 
        # CLIP for visual search
        self.visual_model = SentenceTransformer('clip-ViT-B-32', device=self.device)

    def extract_ocr(self, frame_np):
        results = self.reader.readtext(frame_np)
        # Filter for high confidence and meaningful text length
        return [{"text": res[1], "conf": float(res[2])} for res in results if res[2] > 0.5 and len(res[1]) > 2]

    def extract_objects(self, frame_np):
        results = self.object_model(frame_np, verbose=False)[0]
        return [results.names[int(box.cls)] for box in results.boxes if box.conf > 0.5]

    def get_visual_embedding(self, frame_np):
        frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        return self.visual_model.encode(pil_img).tolist()