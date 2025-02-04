import cv2
from inference import get_model
import supervision as sv
from threading import Thread, Lock
from typing import List
from tqdm import tqdm
import os


mutex = Lock()

cap = cv2.VideoCapture("video.mp4")
fps = int(cap.get(cv2.CAP_PROP_FPS))
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

model_car = get_model(model_id="car-detection-nxsxm-yz6pa-t0cjs-ezgxf/1", api_key=os.environ["ROBOFLOW_KEY"])
model_people = get_model(model_id="people-4evn7-fqlf8-d887c/2", api_key=os.environ["ROBOFLOW_KEY"])

models = [model_car, model_people]

def process(frame, model):
    
    result = model.infer(frame)[0]
    detec = sv.Detections.from_inference(result)

    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    labels = [
    f"{class_name} {confidence:.2f}"
    for class_name, confidence
    in zip(detec['class_name'], detec.confidence)
    ]

    mutex.acquire()
    annotated_image = bounding_box_annotator.annotate(scene=frame, detections=detec)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detec, labels=labels)
    mutex.release()
    

out = cv2.VideoWriter(f'result.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (1920,1080))

bar = tqdm(total=length)
count = 0

while(cap.isOpened()):
    
    ret, frame = cap.read()
    if not ret:
        break
    
    count += 1

    thread_car = Thread(target=process, args=(frame, model_car))
    thread_people = Thread(target=process, args=(frame, model_people))

    threads = [thread_car, thread_people]
    
    for x in threads:
        x.start()

    for x in threads:
        x.join()

    out.write(frame)
    bar.update(count)

    #cv2.imshow("frame.jpg", frame)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

out.release()
cap.release()
cv2.destroyAllWindows()
