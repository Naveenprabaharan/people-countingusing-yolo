# FILE: deep_sort/nn_matching.py

import numpy as np

class NearestNeighborDistanceMetric:
    """A very small distance metric using cosine distance and an embedding gallery.
    This is a simplified version for most single-camera use-cases."""
    def __init__(self, metric, matching_threshold, budget=None):
        assert metric in ("cosine",), "Only cosine supported in this minimal implementation"
        self.metric = metric
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}  # track_id -> [embeddings]

    def partial_fit(self, features, targets, active_targets):
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None and len(self.samples[target]) > self.budget:
                self.samples[target] = self.samples[target][-self.budget:]

        # remove lost tracks
        for t in list(self.samples.keys()):
            if t not in active_targets:
                del self.samples[t]

    def distance(self, features, targets):
        # features: (N, dim), targets: list of track ids
        cost_matrix = np.zeros((len(targets), features.shape[0]), dtype=np.float32)
        for i, t in enumerate(targets):
            target_feats = np.asarray(self.samples.get(t, []))
            if target_feats.size == 0:
                cost_matrix[i, :] = 1.0
                continue
            # compute min cosine distance between each feature and all features of the target
            # cosine distance = 1 - (a.b / (|a||b|))
            a = target_feats / (np.linalg.norm(target_feats, axis=1, keepdims=True) + 1e-6)
            b = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-6)
            # pairwise distances
            d = 1. - np.dot(a, b.T)
            cost_matrix[i, :] = d.min(axis=0)
        return cost_matrix
