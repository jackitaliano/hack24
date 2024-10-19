# Given video / image data, preprocess data to only contain person
import cv2
import numpy as np
from ultralytics import YOLO

def inference_image(file_path, debug):
    model = YOLO("yolo11n.pt")
    results = model(file_path)

    for result in results:
        boxes = result.boxes
        bounding_box = boxes.xyxy[0]
        
        image = cv2.imread(file_path)
        x1, y1, x2, y2 = map(int, bounding_box)
        cropped_image = image[y1:y2, x1:x2]

        if debug:
            cv2.imshow("Cropped Image", cropped_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()    
    
        return cropped_image