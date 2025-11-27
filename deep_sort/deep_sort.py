import numpy as np
from deep_sort.sort_utils import *
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from deep_sort.generate_embeddings import EmbeddingExtractor


class DeepSort:
    def __init__(self, model_path="deep_sort/model_weights/mars-small128.pb", max_dist=0.2):
        metric = NearestNeighborDistanceMetric("cosine", max_dist, None)
        self.tracker = Tracker(metric)
        self.extractor = EmbeddingExtractor(model_path)

    def update(self, frame, detections):
        boxes = np.array([det[:4] for det in detections])
        scores = np.array([det[4] for det in detections])

        # Extract appearance embeddings
        features = self.extractor.extract(frame, boxes)

        dets = [Detection(bbox, score, feat)
                for bbox, score, feat in zip(boxes, scores, features)]

        self.tracker.predict()
        self.tracker.update(dets)

        output = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            x1, y1, x2, y2 = track.to_tlbr()
            output.append([x1, y1, x2, y2, track.track_id])

        return np.array(output)
