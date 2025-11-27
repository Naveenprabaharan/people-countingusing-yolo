import cv2
from ultralytics import YOLO
import numpy as np

# Deep SORT imports (use the local deep_sort package)
from deep_sort.detection import Detection as DS_Detection
from deep_sort.tracker import Tracker as DS_Tracker
from deep_sort import nn_matching

# Load YOLO model
model = YOLO("yolov8n.pt")

# Deep SORT metric and tracker configuration
max_cosine_distance = 0.2
nn_budget = 100
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = DS_Tracker(metric)  # other tracker args use defaults (max_age, n_init, ...)

# Counting variables
counter_in = 0
counter_out = 0

# Store last y-center for each ID
last_positions = {}

# Define counting line (horizontal)
count_line_y = 300

# Feature vector size for appearance (use encoder later to replace this)
FEATURE_DIM = 128

cap = cv2.VideoCapture(0)  # Or CCTV stream

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, classes=[0])  # Only person class

    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            detections.append([x1, y1, x2, y2, conf])

    detections = np.array(detections)

    # Convert to Deep SORT Detection objects (tlwh, confidence, feature)
    ds_dets = []
    for det in detections:
        x1, y1, x2, y2, conf = det
        w = x2 - x1
        h = y2 - y1
        tlwh = [float(x1), float(y1), float(w), float(h)]
        # Provide a non-zero normalized dummy feature so cosine distance is valid.
        # Replace with real ReID features by adding an encoder and filling `feature`.
        dummy_feat = np.ones(FEATURE_DIM, dtype=np.float32)
        dummy_feat /= np.linalg.norm(dummy_feat)
        ds_dets.append(DS_Detection(tlwh, float(conf), feature=dummy_feat))

    # Run Deep SORT
    tracker.predict()
    tracker.update(ds_dets)

    # Collect confirmed tracks
    track_items = []
    for track in tracker.tracks:
        # typical Deep SORT Track API: is_confirmed(), time_since_update, to_tlbr(), track_id
        if hasattr(track, "is_confirmed") and not track.is_confirmed():
            continue
        if hasattr(track, "time_since_update") and track.time_since_update > 1:
            continue
        if hasattr(track, "to_tlbr"):
            x1, y1, x2, y2 = map(int, track.to_tlbr())
        else:
            continue
        track_id = int(getattr(track, "track_id", getattr(track, "track_id_", -1)))
        track_items.append((x1, y1, x2, y2, track_id))

    # Draw line
    cv2.line(frame, (0, count_line_y), (frame.shape[1], count_line_y), (0, 255, 255), 2)

    for x1, y1, x2, y2, track_id in track_items:
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Draw tracker box & id
        cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 255, 50), 2)
        cv2.putText(frame, f'ID:{int(track_id)}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Check movement direction
        if track_id in last_positions:
            prev_y = last_positions[track_id]
            # Person moving down → IN
            if prev_y < count_line_y <= cy:
                counter_in += 1
            # Person moving up → OUT
            elif prev_y > count_line_y >= cy:
                counter_out += 1

        last_positions[track_id] = cy

    # Show counters
    cv2.putText(frame, f"In: {counter_in}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"Out: {counter_out}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("People Counter (Deep SORT)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
