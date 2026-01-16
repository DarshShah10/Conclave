from mmagent.src.perception_modules import PerceptionModules
import cv2
import numpy as np

p = PerceptionModules()
dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
cv2.putText(dummy_frame, "TEST OCR", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

print("Testing OCR:", p.extract_ocr(dummy_frame))
print("Testing Objects:", p.extract_objects(dummy_frame))
print("Testing CLIP:", len(p.get_visual_embedding(dummy_frame)))