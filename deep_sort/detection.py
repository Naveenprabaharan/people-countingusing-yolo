# FILE: deep_sort/detection.py

import numpy as np

class Detection:
    """Simple detection wrapper for DeepSORT.

    bbox: [x1, y1, x2, y2]
    confidence: float
    feature: np.ndarray (embedding)
    """
    def __init__(self, tlbr, confidence, feature):
        self.tlbr = np.asarray(tlbr, dtype=np.float32)
        self.confidence = float(confidence)
        self.feature = feature

    def to_tlwh(self):
        x1, y1, x2, y2 = self.tlbr
        w = x2 - x1
        h = y2 - y1
        return np.array([x1, y1, w, h], dtype=np.float32)

    def to_xyah(self):
        x1, y1, x2, y2 = self.tlbr
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2.
        cy = y1 + h / 2.
        return np.array([cx, cy, a if (a:=w / float(h)) else 0., h], dtype=np.float32)
