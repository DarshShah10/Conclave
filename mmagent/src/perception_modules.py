import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer
from PIL import Image
import torch

class PerceptionModules:
    def __init__(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 1. OCR (English)
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        
        # 2. YOLOv10n (Smallest/Fastest for 8GB VRAM)
        self.object_model = YOLO('yolov10n.pt') 
        
        # 3. CLIP (Visual Embeddings)
        # clip-ViT-B-32 is ~300MB, very efficient for laptops
        self.visual_model = SentenceTransformer('clip-ViT-B-32', device=device)
        
        print(f"Perception Modules initialized on {device}")

    def extract_ocr(self, frame_np):
        """Detects and reads text in the frame."""
        # EasyOCR expects BGR (OpenCV default) or RGB
        results = self.reader.readtext(frame_np)
        return [{
            "text": res[1],
            "bbox": [int(x) for x in [res[0][0][0], res[0][0][1], res[0][2][0], res[0][2][1]]],
            "conf": float(res[2])
        } for res in results if res[2] > 0.4]

    def extract_objects(self, frame_np):
        """Detects physical objects (cars, chairs, laptops, etc.)."""
        results = self.object_model(frame_np, verbose=False)[0]
        return [{
            "label": results.names[int(box.cls)],
            "conf": float(box.conf),
            "bbox": [int(x) for x in box.xyxy[0].tolist()]
        } for box in results.boxes if box.conf > 0.4]

    def get_visual_embedding(self, frame_np):
        """Generates a 512-dimension CLIP vector for the whole frame."""
        # Convert OpenCV BGR to RGB for CLIP
        frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        with torch.no_grad():
            embedding = self.visual_model.encode(pil_img)
        return embedding.tolist()