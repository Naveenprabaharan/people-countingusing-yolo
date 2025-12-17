from multiprocessing import Process
from db import init_db
from camera_counter import run_camera

init_db()


IP_CAMERA_URL1 = f"rtsp://admin:Cogn!@2023@192.168.1.74:554/Streaming/channels/102/"
IP_CAMERA_URL2 = f"rtsp://admin:Cogn!@2023@192.168.1.91:554/Streaming/channels/102/"

cam_in = Process(target=run_camera, args=(
    IP_CAMERA_URL1,
    300,
    "IN",
    "ENTRY CAMERA"
))

cam_out = Process(target=run_camera, args=(
    IP_CAMERA_URL2,
    300,
    "OUT",
    "EXIT CAMERA"
))

cam_in.start()
cam_out.start()

cam_in.join()
cam_out.join()
