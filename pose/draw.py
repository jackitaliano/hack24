import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import cv2
import os

def draw_landmarks_on_image(cropped_image, processed_landmarks, dimensions):
    fig, ax = plt.subplots()
    ax.imshow(cropped_image)
    for landmark in processed_landmarks:
        print(landmark)
        x, y = processed_landmarks[landmark][0], processed_landmarks[landmark][1]
        ax.add_patch(Circle((x * dimensions[1], y * dimensions[0]), 2, color='r'))
    
    ax.plot([processed_landmarks["neck"][0] * dimensions[1], processed_landmarks["pelvis"][0] * dimensions[1]],
        [processed_landmarks["neck"][1] * dimensions[0], processed_landmarks["pelvis"][1] * dimensions[0]], 'r-')
    ax.plot([processed_landmarks["neck"][0] * dimensions[1], processed_landmarks["left_shoulder"][0] * dimensions[1]],
        [processed_landmarks["neck"][1] * dimensions[0], processed_landmarks["left_shoulder"][1] * dimensions[0]], 'b-')
    ax.plot([processed_landmarks["neck"][0] * dimensions[1], processed_landmarks["right_shoulder"][0] * dimensions[1]],
        [processed_landmarks["neck"][1] * dimensions[0], processed_landmarks["right_shoulder"][1] * dimensions[0]], 'b-')
    ax.plot([processed_landmarks["neck"][0] * dimensions[1], processed_landmarks["nose"][0] * dimensions[1]],
        [processed_landmarks["neck"][1] * dimensions[0], processed_landmarks["nose"][1] * dimensions[0]], 'r-')
    ax.plot([processed_landmarks["left_shoulder"][0] * dimensions[1], processed_landmarks["left_elbow"][0] * dimensions[1]],
        [processed_landmarks["left_shoulder"][1] * dimensions[0], processed_landmarks["left_elbow"][1] * dimensions[0]], 'b-')
    ax.plot([processed_landmarks["left_elbow"][0] * dimensions[1], processed_landmarks["left_wrist"][0] * dimensions[1]],
        [processed_landmarks["left_elbow"][1] * dimensions[0], processed_landmarks["left_wrist"][1] * dimensions[0]], 'b-')
    ax.plot([processed_landmarks["right_shoulder"][0] * dimensions[1], processed_landmarks["right_elbow"][0] * dimensions[1]],
        [processed_landmarks["right_shoulder"][1] * dimensions[0], processed_landmarks["right_elbow"][1] * dimensions[0]], 'b-')
    ax.plot([processed_landmarks["right_elbow"][0] * dimensions[1], processed_landmarks["right_wrist"][0] * dimensions[1]],
        [processed_landmarks["right_elbow"][1] * dimensions[0], processed_landmarks["right_wrist"][1] * dimensions[0]], 'b-')
    ax.plot([processed_landmarks["pelvis"][0] * dimensions[1], processed_landmarks["right_hip"][0] * dimensions[1]],
        [processed_landmarks["pelvis"][1] * dimensions[0], processed_landmarks["right_hip"][1] * dimensions[0]], 'g-')
    ax.plot([processed_landmarks["pelvis"][0] * dimensions[1], processed_landmarks["left_hip"][0] * dimensions[1]],
        [processed_landmarks["pelvis"][1] * dimensions[0], processed_landmarks["left_hip"][1] * dimensions[0]], 'g-')
    ax.plot([processed_landmarks["right_knee"][0] * dimensions[1], processed_landmarks["right_ankle"][0] * dimensions[1]],
        [processed_landmarks["right_knee"][1] * dimensions[0], processed_landmarks["right_ankle"][1] * dimensions[0]], 'g-')
    ax.plot([processed_landmarks["left_knee"][0] * dimensions[1], processed_landmarks["left_ankle"][0] * dimensions[1]],
        [processed_landmarks["left_knee"][1] * dimensions[0], processed_landmarks["left_ankle"][1] * dimensions[0]], 'g-')
    ax.plot([processed_landmarks["left_hip"][0] * dimensions[1], processed_landmarks["left_knee"][0] * dimensions[1]],
        [processed_landmarks["left_hip"][1] * dimensions[0], processed_landmarks["left_knee"][1] * dimensions[0]], 'g-')
    ax.plot([processed_landmarks["right_hip"][0] * dimensions[1], processed_landmarks["right_knee"][0] * dimensions[1]],
        [processed_landmarks["right_hip"][1] * dimensions[0], processed_landmarks["right_knee"][1] * dimensions[0]], 'g-')
    
    plt.axis('off')
    plt.savefig('output_image.png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    image = cv2.imread('output_image.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    os.remove('output_image.png')

    return image
    