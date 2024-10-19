import cv2
import numpy as np
import matplotlib.pyplot as plt
import pose.pose as pose
import depth.depth as depth
import preprocess.preprocess as preprocess

def main():
    FILE_PATH = "test_data/walk.mp4"

    cap = cv2.VideoCapture(FILE_PATH)

    all_landmarks = []
    all_distance_map

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        print("cropping image")
        cropped_image = preprocess.inference_image(frame, True)

        print("getting pose landmarks")
        landmarks = pose.get_landmarks(cropped_image, True)

        
        print("getting depth")
        landmarks = depth.inference_image(cropped_image, debug, landmarks)

    cap.release()


main()