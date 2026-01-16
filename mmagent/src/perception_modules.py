import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
import base64

class PerceptionModules:
    def __init__(self):
        # Initialize OCR (English)
        self.reader = easyocr.Reader(['en'], gpu=True)
        # Initialize YOLOv10n (Smallest/Fastest)
        self.object_model = YOLO('yolov10n.pt') 

    def extract_ocr(self, frame_np):
        """Extracts text and its location from a frame."""
        results = self.reader.readtext(frame_np)
        ocr_data = []
        for (bbox, text, prob) in results:
            if prob > 0.5:
                ocr_data.append({
                    "text": text,
                    "bbox": [int(x) for x in [bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1]]],
                    "confidence": float(prob)
                })
        return ocr_data

    def extract_objects(self, frame_np):
        """Detects objects like cars, bags, laptops, etc."""
        results = self.object_model(frame_np, verbose=False)[0]
        objects = []
        for box in results.boxes:
            objects.append({
                "label": results.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": [int(x) for x in box.xyxy[0].tolist()]
            })
        return objects