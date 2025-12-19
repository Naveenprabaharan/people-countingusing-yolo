import cv2
import threading
import time

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

    def read(self, max_age=0.4):
        with self.lock:
            if self.frame is None:
                return None
            if time.time() - self.last_ts > max_age:
                return None
            return self.frame.copy()

    def stop(self):
        self.stopped = True
        self.cap.release()
