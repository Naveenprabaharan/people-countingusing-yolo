import cv2
import numpy as np
import time
from db import read_counts

while True:
    total_in, total_out = read_counts()
    occupied = max(0, total_in - total_out)

    frame = np.zeros((300, 500, 3), dtype=np.uint8)

    cv2.putText(frame, f"Total IN : {total_in}", (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"Total OUT: {total_out}", (50, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.putText(frame, f"Occupied : {occupied}", (50, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,0), 3)

    cv2.imshow("Occupancy Dashboard", frame)
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break
