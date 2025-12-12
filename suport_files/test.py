import cv2
import threading
import time

class RTSPVideoStream:
    def __init__(self, url, reconnect_delay=2):
        self.url = url
        self.reconnect_delay = reconnect_delay
        self.frame = None
        self.stopped = False
        self.connected = False

        # Start thread
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def connect(self):
        print("🔄 Connecting to camera...")
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        time.sleep(0.5)

        if not cap.isOpened():
            print("❌ Failed to connect. Retrying...")
            return None

        print("✅ Connected.")
        return cap

    def update(self):
        cap = None

        while not self.stopped:
            if cap is None:
                cap = self.connect()
                if cap is None:
                    time.sleep(self.reconnect_delay)
                    continue

            ret, frame = cap.read()
            if not ret:
                print("⚠️ Frame lost. Reconnecting...")
                cap.release()
                cap = None
                time.sleep(self.reconnect_delay)
                continue

            self.frame = frame
            self.connected = True

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.thread.join()


# ----------------------------------------------------------
# Example Usage
# ----------------------------------------------------------
if __name__ == "__main__":

    suffix = "192.168.1.41:554/Streaming/channels/102/"
    IP_CAMERA_URL = f"rtsp://admin:Cogn!@2023@{suffix}?rtsp_transport=tcp"

    stream = RTSPVideoStream(IP_CAMERA_URL)

    while True:
        frame = stream.read()
        if frame is not None:
            cv2.imshow("RTSP Stream (Threaded)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    stream.stop()
    cv2.destroyAllWindows()


# import cv2

# suffix = "192.168.1.41:554/Streaming/channels/102/"
# IP_CAMERA_URL = f"rtsp://admin:Cogn!@2023@{suffix}"

# # ----------------------------------------------
# # Better RTSP handling using OpenCV + FFmpeg
# # ----------------------------------------------
# gst = (
#     f"rtspsrc location={IP_CAMERA_URL} latency=0 ! "
#     "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink"
# )

# # Try with GStreamer first (more stable)
# use_gst = False   # Set True if GStreamer installed

# if use_gst:
#     cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
# else:
#     # FFmpeg config for stable RTSP reading
#     cap = cv2.VideoCapture(IP_CAMERA_URL, cv2.CAP_FFMPEG)
#     cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)          # Drop old frames
#     cap.set(cv2.CAP_PROP_FPS, 25)                # Helps some cameras
#     cap.set(cv2.CAP_PROP_POS_FRAMES, 1)

# if not cap.isOpened():
#     print("❌ Cannot open RTSP stream")
#     exit()

# print("📡 Connected to RTSP. Press 'q' to quit.")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("⚠️ Frame read failed — camera skipped frame")
#         continue

#     cv2.imshow("RTSP Camera", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
