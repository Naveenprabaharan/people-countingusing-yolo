# FILE: deep_sort/tracker.py

import numpy as np
from .sort_utils import iou
from .linear_assignment import linear_assignment


class Track:
    def __init__(self, mean, covariance, track_id, n_init=3, max_age=30, initial_feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 0
        self.age = 0
        self.time_since_update = 0
        self.state = 1  # 1 = Tentative, 2 = Confirmed
        self.n_init = n_init
        self.max_age = max_age
        # store appearance features seen for this track (most recent last)
        self.features = []
        if initial_feature is not None:
            # ensure numpy array
            self.features.append(np.asarray(initial_feature))

    def to_tlbr(self):
        # mean contains tlbr in this minimal implementation
        return self.mean

    def predict(self):
        # Minimal implementation: no motion model, so just age the track
        self.age += 1
        self.time_since_update += 1

    def update(self, detection):
        self.mean = detection.tlbr
        self.time_since_update = 0
        self.hits += 1
        if self.state == 1 and self.hits >= self.n_init:
            self.state = 2  # confirmed
        # persist the appearance feature for this track
        if hasattr(detection, 'feature') and detection.feature is not None:
            self.features.append(np.asarray(detection.feature))

    def is_confirmed(self):
        return self.state == 2


class Tracker:
    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.tracks = []
        self._next_id = 1

    def predict(self):
        for t in self.tracks:
            t.predict()

    def update(self, detections):
        # detections: list of Detection objects
        # 1) Compute cost matrix (appearance + iou fallback)
        if len(self.tracks) == 0:
            unmatched_dets = list(range(len(detections)))
            matches = []
            unmatched_tracks = []
        else:
            # prepare detection features and track ids for appearance distance
            if len(detections) > 0:
                det_features = np.array([d.feature for d in detections])
            else:
                det_features = np.zeros((0, 128), dtype=np.float32)

            # FALLBACK: if metric has no samples yet or detections have no features,
            # use IOU-based assignment instead of appearance distance
            if len(self.metric.samples) == 0 or det_features.size == 0:
                # compute IOU matrix between existing tracks and detections
                trk_boxes = np.array([t.to_tlbr() for t in self.tracks])
                iou_matrix = np.zeros((len(trk_boxes), len(detections)), dtype=np.float32)
                for t, trk_box in enumerate(trk_boxes):
                    for d, det in enumerate(detections):
                        iou_matrix[t, d] = iou(trk_box, det.tlbr)

                # cost = 1 - iou; accept matches with iou > self.max_iou_distance
                cost_matrix = 1.0 - iou_matrix
                matches, u_tracks, u_dets = linear_assignment(cost_matrix, 1.0 - self.max_iou_distance)
                unmatched_tracks = list(u_tracks)
                unmatched_dets = list(u_dets)
            else:
                track_ids = [t.track_id for t in self.tracks]
                cost_matrix = self.metric.distance(det_features, track_ids)

                matches, u_tracks, u_dets = linear_assignment(cost_matrix, self.metric.matching_threshold)

                unmatched_tracks = list(u_tracks)
                unmatched_dets = list(u_dets)

                # now verify matched pairs with IOU threshold; if IOU low, mark as unmatched
                verified_matches = []
                for r, c in matches:
                    trk = self.tracks[r]
                    det = detections[c]
                    if iou(trk.to_tlbr(), det.tlbr) > self.max_iou_distance:
                        verified_matches.append([r, c])
                    else:
                        unmatched_tracks.append(r)
                        unmatched_dets.append(c)
                matches = verified_matches

        # 2) Update matched tracks
        for r, c in matches:
            self.tracks[r].update(detections[c])

        # 3) Create new tracks for unmatched detections
        for idx in unmatched_dets:
            det = detections[idx]
            mean = det.tlbr
            cov = np.eye(4)
            # create new track and initialize it with the detection so hits/features/time_since_update are set
            new_track = Track(mean, cov, self._next_id, n_init=self.n_init, max_age=self.max_age)
            # initialize / confirm template: this also appends the detection.feature to the track
            new_track.update(det)
            self._next_id += 1
            self.tracks.append(new_track)

        # 4) Age and remove old tracks
        new_tracks = []
        for t in self.tracks:
            if t.time_since_update > t.max_age:
                continue
            new_tracks.append(t)
        self.tracks = new_tracks

        # 5) Update metric with confirmed tracks
        features = []
        targets = []
        # active targets should include all currently tracked ids (including tentative) to avoid premature deletion
        active_targets = [t.track_id for t in self.tracks]

        for t in self.tracks:
            if t.is_confirmed() and len(t.features) > 0:
                # use most recent feature for this confirmed track
                features.append(t.features[-1])
                targets.append(t.track_id)

        if len(features) > 0:
            try:
                feats = np.asarray(features)
                self.metric.partial_fit(feats, targets, active_targets)
            except Exception:
                # metric updates are best-effort in this minimal implementation
                pass