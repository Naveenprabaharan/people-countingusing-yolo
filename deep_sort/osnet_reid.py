import os
import numpy as np
import cv2

try:
    import torch
    from torchvision import transforms
    # try to import torchreid (preferred)
    from torchreid.models import build_model
    from torchreid.utils import load_pretrained_weights

    class OSNetReID:
        """OSNet wrapper using torchreid. Provide model_path to load weights."""
        def __init__(self, model_path=None, model_name="osnet_x1_0", device=None):
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model = build_model(
                name=model_name,
                num_classes=1000,
                loss="softmax",
                pretrained=False
            )
            if model_path is not None and os.path.exists(model_path):
                load_pretrained_weights(self.model, model_path)
            self.model = self.model.to(self.device)
            self.model.eval()
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            self.feat_dim = self._infer_feat_dim()

        def _infer_feat_dim(self):
            with torch.no_grad():
                x = torch.zeros(1, 3, 256, 128, device=self.device)
                out = self.model(x)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                return int(out.shape[1]) if hasattr(out, 'shape') else 512

        def extract(self, frame, tlwh_boxes):
            crops = []
            for tlwh in tlwh_boxes:
                x, y, w, h = map(int, tlwh)
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = frame[y1:y2, x1:x2][:, :, ::-1]  # BGR->RGB
                inp = self.transform(crop)
                crops.append(inp)
            if len(crops) == 0:
                return np.zeros((0, self.feat_dim), dtype=np.float32)
            batch = torch.stack(crops, dim=0).to(self.device)
            with torch.no_grad():
                out = self.model(batch)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                feats = out.cpu().numpy()
            norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12
            feats = feats.astype(np.float32) / norms.astype(np.float32)
            return feats

except Exception:
    # Fallback lightweight extractor (HSV hist) when torchreid/torch not available.
    class OSNetReID:
        """
        Fallback: histogram-based feature extractor with same API as OSNetReID.
        Returns L2-normalized vectors of length feat_dim.
        """
        def __init__(self, model_path=None, model_name="fallback_hist", device=None, feat_dim=128):
            self.feat_dim = feat_dim

        def extract(self, frame, tlwh_boxes):
            feats = []
            for tlwh in tlwh_boxes:
                x, y, w, h = map(int, tlwh)
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
                if x2 <= x1 or y2 <= y1:
                    hist = np.zeros((self.feat_dim,), dtype=np.float32)
                    feats.append(hist)
                    continue
                crop = frame[y1:y2, x1:x2]
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                # Use 2D HS histogram and flatten
                h_bins = max(8, self.feat_dim // 8)
                s_bins = max(8, self.feat_dim // 8)
                hist = cv2.calcHist([hsv], [0, 1], None, [h_bins, s_bins], [0, 180, 0, 256])
                hist = hist.flatten().astype(np.float32)
                if hist.sum() > 0:
                    hist = hist / (np.linalg.norm(hist) + 1e-6)
                # pad/trim
                if hist.size < self.feat_dim:
                    hist = np.pad(hist, (0, self.feat_dim - hist.size), mode='constant')
                else:
                    hist = hist[:self.feat_dim]
                feats.append(hist)
            if len(feats) == 0:
                return np.zeros((0, self.feat_dim), dtype=np.float32)
            return np.vstack(feats).astype(np.float32)