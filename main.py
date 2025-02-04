import cv2
from inference import get_model
from threading import Thread, Lock
import subprocess
from tqdm import tqdm
import os
import shutil
from IdentifyThread import IdentifyThread


NUMBER_OF_THREADS = 4
ROBOFLOW_KEY = os.environ["ROBOFLOW_KEY"]

if(shutil.os.path.exists("seg")):
  shutil.rmtree("seg")
shutil.os.mkdir("seg")
if(shutil.os.path.exists("out")):
  shutil.rmtree("out")
shutil.os.mkdir("out")

mutex = Lock()

cap = cv2.VideoCapture("video.mp4")
fps = int(cap.get(cv2.CAP_PROP_FPS))
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

time_seconds = length // fps + 1

cmd = [
    "ffmpeg",
    "-i", "video.mp4",
    "-c", "copy",
    "-map", "0",
    "-segment_time", str(time_seconds / NUMBER_OF_THREADS),
    "-f", "segment",
    "-reset_timestamps", "1",
    "seg/segment-%02d.mp4"
]

subprocess.run(cmd)

modelDesc = {
   "car" :  "car-detection-nxsxm-yz6pa-t0cjs-ezgxf/1",
   "people" : "people-4evn7-fqlf8-d887c/2",
   "key" : ROBOFLOW_KEY
}

videos = [f"seg/{x}" for x in os.listdir("seg/")]
videos = sorted(videos, key=lambda x: int(x.split('-')[-1].split('.')[0]))

mainTreads = []

for idx, v in enumerate(videos):
    mainTreads.append(IdentifyThread(v, idx, modelDesc))

for t in mainTreads:
    t.start()

for t in mainTreads:
    t.join()

print("Mergin results")

with open("output.txt", "w") as outfile:
    for filename in sorted(os.listdir("out")):
        outfile.write(f"file 'out/{filename}'\n")

cmd = [
   "ffmpeg",
    "-f", "concat",
    "-i", "output.txt",
    "-c", "copy",
    "result.avi"
]

subprocess.run(cmd)