# FILE: example_main_integration.py

"""Example integration showing how to use the minimal deep_sort package with YOLO (ultralytics).

This file is provided as a helper. It is NOT repeated in chat. Use it after saving the package files.
"""

# USAGE (after creating deep_sort/ module and placing mars-small128.pb):
# 1) pip install ultralytics filterpy scipy pillow tensorflow
# 2) python example_main_integration.py --source 0

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort.generate_embeddings import EmbeddingExtractor
from deep_sort.detection import Detection
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.tracker import Tracker


def run(source='video/Walking.mp4', model_path='yolov8n.pt', reid_path='deep_sort/model_weights/mars-small128.pb'):
    model = YOLO(model_path)
    extractor = EmbeddingExtractor(reid_path)
    # raise matching_threshold and reduce n_init to help stabilize ids during debugging
    metric = NearestNeighborDistanceMetric('cosine', matching_threshold=0.6, budget=100)
    tracker = Tracker(metric, max_iou_distance=0.7, max_age=30, n_init=1)

    cap = cv2.VideoCapture(source)
    count_in = 0
    count_out = 0
    last_positions = {}
    frame_h = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_h is None:
            frame_h = frame.shape[0]
        results = model(frame, classes=[0])
        
        # print('results : ',results)
        
        dets = []
        for r in results:
            for box in r.boxes:
                # print(f'box: {box.xyxy[0].cpu().numpy()}')
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                dets.append([x1, y1, x2, y2, conf])
                # print([x1, y1, x2, y2, conf])
        dets = np.array(dets)
        # print('dets : ',dets)

        # Prepare DeepSORT detections
        boxes = dets[:, :4] if dets.size else np.empty((0,4))
        scores = dets[:, 4] if dets.size else np.empty((0,))
        if boxes.shape[0] > 0:
            features = extractor.extract(frame, boxes)
            detections = [Detection(tlbr=boxes[i], confidence=scores[i], feature=features[i]) for i in range(len(boxes))]
        else:
            detections = []

        tracker.predict()
        tracker.update(detections)

        # draw
        count_line = frame_h // 2
        cv2.line(frame, (0, count_line), (frame.shape[1], count_line), (0,255,255), 2)

        for trk in tracker.tracks:
            # print(f'trk.time_since_update:{trk.time_since_update}')
            # print(f'trk.trk.is_confirmed():{trk.is_confirmed()}')
            if not trk.is_confirmed() or trk.time_since_update > 2:
                continue
            x1, y1, x2, y2 = trk.to_tlbr()
            # print(f'data: {x1, y1, x2, y2}')
            tid = trk.track_id
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(frame, f'ID:{tid}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)
            print(f'fid:{tid}')
            # print(f'put rect')
            if tid in last_positions:
                prev_y = last_positions[tid]
                if prev_y < count_line <= cy:
                    count_in += 1
                elif prev_y > count_line >= cy:
                    count_out += 1
            last_positions[tid] = cy

        cv2.putText(frame, f'In: {count_in}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.putText(frame, f'Out: {count_out}', (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow('DeepSORT demo', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()