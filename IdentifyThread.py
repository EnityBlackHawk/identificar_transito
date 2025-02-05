from threading import Thread, Lock
import cv2
from tqdm import tqdm
import supervision as sv
from inference import get_model

class IdentifyThread(Thread):
    def __init__(self, video, segment, modeldesc):
        super().__init__()
        self.video = video
        self.segment = segment
        self.mutex = Lock()
        self.model_car =  get_model(modeldesc["car"], modeldesc["key"])
        self.model_people = get_model(modeldesc["people"], modeldesc["key"])
        self.model_signs = get_model(modeldesc["signs"], modeldesc["key"])
    
    def processFrame(self, frame, model, color):
    
        result = model.infer(frame)[0]
        detec = sv.Detections.from_inference(result)
        detec = detec[detec.confidence > 0.70]

        bounding_box_annotator = sv.BoxAnnotator(color=color)
        label_annotator = sv.LabelAnnotator(color=color)

        labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(detec['class_name'], detec.confidence)
        ]

        self.mutex.acquire()
        annotated_image = bounding_box_annotator.annotate(scene=frame, detections=detec)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detec, labels=labels)
        self.mutex.release()

    def run(self):
        segCap = cv2.VideoCapture(self.video)
        outName = f"out/{self.segment}.avi"
        out = cv2.VideoWriter(outName,cv2.VideoWriter_fourcc('M','J','P','G'), 30, (1920,1080))
        segLen = segCap.get(cv2.CAP_PROP_FRAME_COUNT)
        bar = tqdm(total=segLen)
        count = 0

        while(segCap.isOpened()):
            
            ret, frame = segCap.read()
            if not ret:
                break
            
            count += 1

            thread_car = Thread(target=self.processFrame, args=(frame, self.model_car, sv.Color.BLUE))
            thread_people = Thread(target=self.processFrame, args=(frame, self.model_people, sv.Color.ROBOFLOW))
            thread_sign = Thread(target=self.processFrame, args=(frame, self.model_signs, sv.Color.RED))

            threads = [thread_car, thread_people, thread_sign]
            
            for x in threads:
                x.start()

            for x in threads:
                x.join()

            out.write(frame)
            bar.update(count)
        out.release()