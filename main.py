import cv2
from ultralytics import YOLO
from utils.sort import Sort  # Import SORT tracker

# Load YOLO model
model = YOLO("yolov8n.pt")

# Initialize tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Counting variables
counter_in = 0
counter_out = 0

# Store last y-center for each ID
last_positions = {}

# Define counting line (horizontal)
count_line_y = 300

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

    # Convert to numpy array for SORT
    import numpy as np
    detections = np.array(detections)

    # Track objects
    tracks = tracker.update(detections)

    # Draw line
    cv2.line(frame, (0, count_line_y), (frame.shape[1], count_line_y), (0, 255, 255), 2)

    for track in tracks:
        x1, y1, x2, y2, track_id = track.astype(int)

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Draw tracker box & id
        cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 255, 50), 2)
        cv2.putText(frame, f'ID:{int(track_id)}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Check movement direction
        if track_id in last_positions:
            prev_y = last_positions[track_id]
            delta = cy - prev_y

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

    cv2.imshow("People Counter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
