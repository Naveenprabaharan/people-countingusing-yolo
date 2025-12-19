import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort.tracker import Tracker
from deep_sort import nn_matching
from deep_sort.detection import Detection
from db import increment_in, increment_out
import time, threading

FEATURE_DIM = 128


class RTSPStream:
    def __init__(self, url):
        self.url = url
        self.frame = None
        self.last_ts = 0
        self.lock = threading.Lock()
        self.stopped = False

        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
                    self.last_ts = time.time()
            else:
                self.cap.release()
                time.sleep(1)
                self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self, max_age=0.3):
        with self.lock:
            if self.frame is None:
                return None
            if time.time() - self.last_ts > max_age:
                return None
            return self.frame.copy()

    def stop(self):
        self.stopped = True
        self.cap.release()



def run_camera(
    rtsp_url,
    count_line_x,
    direction,        # "IN" or "OUT"
    window_name
):
    model = YOLO("yolo12n.pt")

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.2, 100)
    # tracker = Tracker(metric)
    tracker = Tracker(
    metric,
    max_age=30,     # allow missed frames
    n_init=2
    )

    last_positions = {}

    # cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    stream = RTSPStream(rtsp_url)
    while True:
        frame = stream.read()
        if frame is None:
            continue

        results = model(frame, classes=[0], verbose=False)
        detections = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                detections.append([x1, y1, x2, y2, conf])

        ds_dets = []
        for x1, y1, x2, y2, conf in detections:
            tlwh = [x1, y1, x2 - x1, y2 - y1]
            feat = np.ones(FEATURE_DIM, dtype=np.float32)
            feat /= np.linalg.norm(feat)
            ds_dets.append(Detection(tlwh, conf, feat))

        tracker.predict()
        tracker.update(ds_dets)

        cv2.line(frame, (count_line_x, 0), (count_line_x, frame.shape[0]), (0,255,255), 2)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            x1, y1, x2, y2 = map(int, track.to_tlbr())
            track_id = track.track_id
            cx = (x1 + x2) // 2

            if track_id in last_positions:
                prev_x = last_positions[track_id]

                if direction == "IN" and prev_x < count_line_x <= cx:
                    increment_in()
                elif direction == "OUT" and prev_x > count_line_x >= cx:
                    increment_out()

            last_positions[track_id] = cx

            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,f"ID:{track_id}",(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        cv2.imshow(window_name, cv2.resize(frame,(640,480)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # cap.release()
