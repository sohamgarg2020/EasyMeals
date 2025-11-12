import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
import numpy as np
from collections import Counter




FOOD_CLASSES = ['apple','banana','orange','broccoli','carrot','sandwich','pizza','cake','donut','hot dog','bowl','cup','wine glass','spoon','fork','knife', 'bottle']


model = YOLO('yolov8m.pt')



def ingredient_list(img_path):

    img = cv2.imread(img_path)
    img = cv2.convertScaleAbs(img, alpha=1.2, beta=20)

    results = model(img, conf = 0.15, imgsz=1024)


    annotated_frame = results[0].plot()

    cv2.imshow("Detections:", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    detected = Counter([model.names[int(box.cls)] for r in results for box in r.boxes])

    for key in list(detected.keys()):
        if key not in FOOD_CLASSES:
            del detected[key]

    print(detected)
    return detected


