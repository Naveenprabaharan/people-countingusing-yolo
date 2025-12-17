import cv2

url = "rtsp://100.98.252.74:8554/live"
# 100.98.252.74
cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run ML inference here

    cv2.imshow("People Counter (Deep SORT)", cv2.resize(frame,(640,480)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()