import os
import urllib.request

if not os.path.exists("osnet_x0_25_msmt17.pt"):
    url = "https://huggingface.co/paulosantiago/osnet_x0_25_msmt17/resolve/main/osnet_x0_25_msmt17.pt?download=true"
    urllib.request.urlretrieve(url, "osnet_x0_25_msmt17.pt")
    print("ReID model download complete")
else:
    print("ReID model already exists")

from ultralytics import YOLO

if not os.path.exists('yolov8s.pt'):
    YOLO('yolov8s.pt')  # Ultralytics will auto-download it
    print("YOLO model downloaded")
else:
    print("YOLO model already exists")