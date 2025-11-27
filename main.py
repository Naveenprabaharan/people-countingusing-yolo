import cv2
from ultralytics import YOLO
import numpy as np

# Deep SORT imports (use the local deep_sort package)
from deep_sort.detection import Detection as DS_Detection
from deep_sort.tracker import Tracker as DS_Tracker
from deep_sort import nn_matching

# New: OSNet ReID extractor
from deep_sort.osnet_reid import OSNetReID

# Load YOLO model
model = YOLO("yolov8n.pt")

# Instantiate ReID extractor (provide model_path if you have a local osnet .pth)
# If you don't have weights, install torchreid and download OSNet weights or pass a path.
reid_model_path = './deep_sort/osnet_model/osnet_x1_0_imagenet.pth'  # e.g. "deep_sort/osnet_x1_0_pretrained.pth"
extractor = OSNetReID(model_path=reid_model_path)
FEATURE_DIM = extractor.feat_dim

# Deep SORT metric and tracker configuration (tuned)
max_cosine_distance = 0.45   # loosened for robustness; tune between 0.3..0.6
nn_budget = 150              # keep more embeddings per ID
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
# tune tracker lifecycle: n_init (confirm), max_age (occlusion tolerance), max_iou_distance
tracker = DS_Tracker(metric, max_iou_distance=0.7, max_age=30, n_init=2)

# Counting variables
counter_in = 0
counter_out = 0

# Store last x-center for each ID (for vertical line counting)
last_positions = {}

# Define counting line (vertical)
count_line_x = 300  # adjust this x coordinate as needed

cap = cv2.VideoCapture('video/Walking.mp4')  # Or CCTV stream

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

    # Prepare boxes for ReID extractor (tlwh)
    tlwh_boxes = []
    for det in detections:
        x1, y1, x2, y2, conf = det
        w = x2 - x1
        h = y2 - y1
        tlwh_boxes.append([float(x1), float(y1), float(w), float(h)])

    # Run ReID extractor in batch
    if len(tlwh_boxes) > 0:
        reid_feats = extractor.extract(frame, tlwh_boxes)  # NxFEATURE_DIM, L2-normalized
    else:
        reid_feats = np.zeros((0, FEATURE_DIM), dtype=np.float32)

    # Convert to Deep SORT Detection objects (tlwh, confidence, feature)
    ds_dets = []
    for i, det in enumerate(detections):
        x1, y1, x2, y2, conf = det
        w = x2 - x1
        h = y2 - y1
        tlwh = [float(x1), float(y1), float(w), float(h)]
        # match features by index; if extractor skipped invalid boxes, lengths should still match.
        if i < reid_feats.shape[0]:
            feat = reid_feats[i]
        else:
            feat = np.ones(FEATURE_DIM, dtype=np.float32)
            feat /= np.linalg.norm(feat)
        ds_dets.append(DS_Detection(tlwh, float(conf), feature=feat))

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

    # Draw vertical counting line
    cv2.line(frame, (count_line_x, 0), (count_line_x, frame.shape[0]), (0, 255, 255), 2)

    for x1, y1, x2, y2, track_id in track_items:
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Draw tracker box & id
        cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 255, 50), 2)
        cv2.putText(frame, f'ID:{int(track_id)}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Check movement direction across vertical line (use x center)
        if track_id in last_positions:
            prev_x = last_positions[track_id]
            # Person moving right → IN
            if prev_x < count_line_x <= cx:
                counter_in += 1
            # Person moving left → OUT
            elif prev_x > count_line_x >= cx:
                counter_out += 1

        last_positions[track_id] = cx

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
