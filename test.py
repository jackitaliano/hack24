import cv2
import numpy as np
import matplotlib.pyplot as plt
import pose.pose as pose
import depth.depth as depth
import preprocess.preprocess as preprocess
from pose import draw

def main():
    FILE_PATH = "test_data/walk.mp4"

    cap = cv2.VideoCapture(FILE_PATH)

    all_landmarks = []
    all_distance_map = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        print("cropping image")
        cropped_image = preprocess.inference_image(frame, True)

        print("getting pose landmarks")
        landmarks = pose.get_landmarks(cropped_image, True)
        
        annotated_img = np.copy(landmarks)
        for keypoint in landmarks:
            cv2.circle(annotated_img, (landmarks[keypoint][0], landmarks[keypoint][1]), 5, (0, 255, 0), -1)

        all_landmarks.append(annotated_img)

    cap.release()


main()