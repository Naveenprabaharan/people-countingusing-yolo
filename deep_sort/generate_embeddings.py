# FILE: deep_sort/generate_embeddings.py

import numpy as np
from PIL import Image
import tensorflow as tf

class EmbeddingExtractor:
    """A tiny wrapper around a TF frozen graph (mars-small128.pb) to extract 128-d features.

    This implementation loads the pb into a TF graph and runs it. It's intentionally
    minimal and may be slower than optimized alternatives.
    """
    def __init__(self, model_path):
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(self.model_path, 'rb') as f:
            graph_def.ParseFromString(f.read())
        g = tf.Graph()
        with g.as_default():
            tf.import_graph_def(graph_def, name='')
        self.sess = tf.compat.v1.Session(graph=g)
        # heuristics for common input/output names from mars-small128
        self.input_tensor = g.get_tensor_by_name('input:0') if 'input:0' in [t.name for t in g.as_graph_def().node] else None
        # try several common names
        try:
            self.input_tensor = g.get_tensor_by_name('images:0')
        except Exception:
            pass
        # output
        try:
            self.output_tensor = g.get_tensor_by_name('features:0')
        except Exception:
            # fallback: take the last tensor
            self.output_tensor = [n for n in g.as_graph_def().node if 'bias' not in n.name][-1]
            # we won't be able to fetch by node; this is a best-effort minimal loader
            self.output_tensor = self.sess.graph.get_operations()[-1].outputs[0]

    def _preprocess(self, frame, boxes):
        crops = []
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                crop = np.zeros((64, 128, 3), dtype=np.uint8)
            img = Image.fromarray(crop).resize((64, 128))
            arr = np.asarray(img).astype(np.float32)
            arr = (arr - 127.5) / 128.0
            crops.append(arr)
        if len(crops) == 0:
            return np.zeros((0, 128), dtype=np.float32), []
        batch = np.stack(crops, axis=0)
        return batch, boxes

    def extract(self, frame, boxes):
        # boxes: np.array Nx4 tlbr
        batch, boxes = self._preprocess(frame, boxes)
        if batch.shape[0] == 0:
            return np.zeros((0, 128), dtype=np.float32)
        feed = {self.input_tensor: batch}
        feats = self.sess.run(self.output_tensor, feed_dict=feed)
        feats = np.asarray(feats)
        # L2-normalize
        norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-6
        feats = feats / norms
        return feats
