

# ================= CONFIG =================


ENTRY_RTSP = (
    "rtsp://admin:Cogn!@2023@192.168.1.74:554/Streaming/channels/102/"#rtsp://admin:Cogn!@2023@192.168.1.74/Streaming/channels/102
    "?rtsp_transport=tcp&fflags=nobuffer&flags=low_delay&max_delay=0"
)
EXIT_RTSP = (
    "rtsp://admin:Cogn!@2023@192.168.1.61:554/Streaming/channels/102/"
    "?rtsp_transport=tcp&fflags=nobuffer&flags=low_delay&max_delay=0"
)


# =========================================
import cv2
import time
from ultralytics import YOLO
from db import init_db, increment_in, increment_out
from rtsp_stream import RTSPStream

# ================= CONFIG =================

COUNT_LINE_X = 300
FRAME_SKIP = 2
# =========================================

init_db()

# 🔥 ONE YOLO MODEL
model = YOLO("yolo12n.pt")

entry_last_pos = {}
exit_last_pos = {}

entry_stream = RTSPStream(ENTRY_RTSP)
exit_stream  = RTSPStream(EXIT_RTSP)

frame_id = 0

def process_camera(frame, last_pos, direction, window):
    frame = cv2.resize(frame, (640, 384))

    # 🔥 ByteTrack enabled automatically
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        verbose=False
    )

    cv2.line(frame, (COUNT_LINE_X, 0),
             (COUNT_LINE_X, frame.shape[0]), (0,255,255), 2)

    boxes = results[0].boxes
    # if boxes is None:
    #     cv2.imshow(window, frame)
    #     return

    for box in boxes:
        if box.id is not None:
            tid = int(box.id[0])
        else: continue

        x1,y1,x2,y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2

        if tid in last_pos:
            prev = last_pos[tid]
            if direction == "IN" and prev < COUNT_LINE_X <= cx:
                increment_in()
            elif direction == "OUT" and prev > COUNT_LINE_X >= cx:
                increment_out()

        last_pos[tid] = cx

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame,f"ID:{tid}",(x1,y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    cv2.imshow(window, frame)

# ================= MAIN LOOP =================
while True:
    entry_frame = entry_stream.read()
    exit_frame  = exit_stream.read()

    frame_id += 1
    if frame_id % FRAME_SKIP != 0:
        # if entry_frame is not None:
        #     cv2.imshow("ENTRY", entry_frame)
        # if exit_frame is not None:
        #     cv2.imshow("EXIT", exit_frame)
        cv2.waitKey(1)
        continue

    if entry_frame is not None:
        process_camera(entry_frame, entry_last_pos, "IN", "ENTRY")

    if exit_frame is not None:
        process_camera(exit_frame, exit_last_pos, "OUT", "EXIT")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

entry_stream.stop()
exit_stream.stop()
cv2.destroyAllWindows()
