import argparse
import preprocess.preprocess as preprocess
import depth.depth as depth
import cv2
import matplotlib.pyplot as plt
from pose import pose, process_data, draw
from matplotlib.patches import Circle
import json
import numpy as np


TRUTH_PATH = "test_data/extend_arms.json"

def normalize_landmarks(landmarks):

    for keypoint in landmarks:
        landmarks[keypoint][1] = 1 - landmarks[keypoint][1]
        landmarks[keypoint][2] *= -1

    return landmarks


def process_frame(cropped_image, debug):
    dimensions = cropped_image.shape

    print("getting pose landmarks")
    landmarks = pose.get_landmarks(cropped_image, debug)

    landmarks_results = pose.get_landmarks(cropped_image, debug)

    np_landmarks = process_data.get_np_landmarks(landmarks_results.pose_landmarks)

    processed_landmarks = process_data.get_processed_landmarks(np_landmarks)

    if debug:
        print("drawing landmarks")
        annotated_image = draw.draw_landmarks_on_image(
            cropped_image, processed_landmarks, dimensions
        )
        plt.imshow(annotated_image)
        plt.show()

    print("getting depth")
    landmarks = depth.inference_image(cropped_image, debug, processed_landmarks)

    landmarks = normalize_landmarks(processed_landmarks)

    return landmarks


def display_pose_3d(landmarks, color, ax, zorder):

    ax.scatter(landmarks["nose"][0], landmarks["nose"][2], landmarks["nose"][1], c=color, marker="o", s=250, zorder=zorder)  # s is the size of the marker

    ax.scatter(landmarks["left_shoulder"][0], landmarks["left_shoulder"][2], landmarks["left_shoulder"][1], c=color, marker="o", s=100,zorder=zorder)
    ax.scatter(landmarks["right_shoulder"][0], landmarks["right_shoulder"][2], landmarks["right_shoulder"][1], c=color, marker="o", s=100, zorder=zorder)

    ax.scatter(landmarks["left_elbow"][0], landmarks["left_elbow"][2], landmarks["left_elbow"][1], c=color, marker="o", s=75, zorder=zorder)
    ax.scatter(landmarks["right_elbow"][0], landmarks["right_elbow"][2], landmarks["right_elbow"][1], c=color, marker="o", s=75, zorder=zorder)

    ax.scatter(landmarks["left_wrist"][0], landmarks["left_wrist"][2], landmarks["left_wrist"][1], c=color, marker="o", s=75, zorder=zorder)
    ax.scatter(landmarks["right_wrist"][0], landmarks["right_wrist"][2], landmarks["right_wrist"][1], c=color, marker="o", s=75, zorder=zorder)

    ax.scatter(landmarks["left_hip"][0], landmarks["left_hip"][2], landmarks["left_hip"][1], c=color, marker="o", s=100, zorder=zorder)
    ax.scatter(landmarks["right_hip"][0], landmarks["right_hip"][2], landmarks["right_hip"][1], c=color, marker="o", s=100, zorder=zorder)

    ax.scatter(landmarks["left_knee"][0], landmarks["left_knee"][2], landmarks["left_knee"][1], c=color, marker="o", s=75, zorder=zorder)
    ax.scatter(landmarks["right_knee"][0], landmarks["right_knee"][2], landmarks["right_knee"][1], c=color, marker="o", s=75, zorder=zorder)

    ax.scatter(landmarks["left_ankle"][0], landmarks["left_ankle"][2], landmarks["left_ankle"][1], c=color, marker="o", s=75, zorder=zorder)
    ax.scatter(landmarks["right_ankle"][0], landmarks["right_ankle"][2], landmarks["right_ankle"][1], c=color, marker="o", s=75, zorder=zorder)

    ax.plot([landmarks["nose"][0], landmarks["neck"][0]], [landmarks["nose"][2], landmarks["neck"][2]], [landmarks["nose"][1], landmarks["neck"][1]], c=color, linewidth=5, zorder=zorder)
    ax.plot([landmarks["neck"][0], landmarks["left_shoulder"][0]], [landmarks["neck"][2], landmarks["left_shoulder"][2]], [landmarks["neck"][1], landmarks["left_shoulder"][1]], c=color, linewidth=5, zorder=zorder)
    ax.plot([landmarks["neck"][0], landmarks["right_shoulder"][0]], [landmarks["neck"][2], landmarks["right_shoulder"][2]], [landmarks["neck"][1], landmarks["right_shoulder"][1]], c=color, linewidth=5, zorder=zorder)
    ax.plot([landmarks["left_shoulder"][0], landmarks["left_elbow"][0]], [landmarks["left_shoulder"][2], landmarks["left_elbow"][2]], [landmarks["left_shoulder"][1], landmarks["left_elbow"][1]], c=color, linewidth=5, zorder=zorder)
    ax.plot([landmarks["right_shoulder"][0], landmarks["right_elbow"][0]], [landmarks["right_shoulder"][2], landmarks["right_elbow"][2]], [landmarks["right_shoulder"][1], landmarks["right_elbow"][1]], c=color, linewidth=5, zorder=zorder)
    ax.plot([landmarks["left_elbow"][0], landmarks["left_wrist"][0]], [landmarks["left_elbow"][2], landmarks["left_wrist"][2]], [landmarks["left_elbow"][1], landmarks["left_wrist"][1]], c=color, linewidth=5, zorder=zorder)
    ax.plot([landmarks["right_elbow"][0], landmarks["right_wrist"][0]], [landmarks["right_elbow"][2], landmarks["right_wrist"][2]], [landmarks["right_elbow"][1], landmarks["right_wrist"][1]], c=color, linewidth=5, zorder=zorder)
    ax.plot([landmarks["neck"][0], landmarks["pelvis"][0]], [landmarks["neck"][2], landmarks["pelvis"][2]], [landmarks["neck"][1], landmarks["pelvis"][1]], c=color, linewidth=5, zorder=zorder)
    ax.plot([landmarks["pelvis"][0], landmarks["left_hip"][0]], [landmarks["pelvis"][2], landmarks["left_hip"][2]], [landmarks["pelvis"][1], landmarks["left_hip"][1]], c=color, linewidth=5, zorder=zorder)
    ax.plot([landmarks["pelvis"][0], landmarks["right_hip"][0]], [landmarks["pelvis"][2], landmarks["right_hip"][2]], [landmarks["pelvis"][1], landmarks["right_hip"][1]], c=color, linewidth=5, zorder=zorder)
    ax.plot([landmarks["left_hip"][0], landmarks["left_knee"][0]], [landmarks["left_hip"][2], landmarks["left_knee"][2]], [landmarks["left_hip"][1], landmarks["left_knee"][1]], c=color, linewidth=5, zorder=zorder)
    ax.plot([landmarks["right_hip"][0], landmarks["right_knee"][0]], [landmarks["right_hip"][2], landmarks["right_knee"][2]], [landmarks["right_hip"][1], landmarks["right_knee"][1]], c=color, linewidth=5, zorder=zorder)
    ax.plot([landmarks["left_knee"][0], landmarks["left_ankle"][0]], [landmarks["left_knee"][2], landmarks["left_ankle"][2]], [landmarks["left_knee"][1], landmarks["left_ankle"][1]], c=color, linewidth=5, zorder=zorder)
    ax.plot([landmarks["right_knee"][0], landmarks["right_ankle"][0]], [landmarks["right_knee"][2], landmarks["right_ankle"][2]], [landmarks["right_knee"][1], landmarks["right_ankle"][1]], c=color, linewidth=5, zorder=zorder)

    ax.set_xlim([0, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 1])

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    return ax

def find_distance(test_landmarks, truth_landmarks):
    score = 0
    for keypoint in test_landmarks:
        for i in range(3):
            score += np.sqrt((test_landmarks[keypoint][i] - truth_landmarks[keypoint][i]) ** 2)
    
    return np.mean(score)

def main(type, file_path, debug, truth):

    if type == "image":
        np_img = plt.imread(file_path)
        landmarks = process_frame(np_img, debug)

        display_pose_3d(landmarks)

        return landmarks

    elif type == "video":
        landmarks_per_frame = []

        cap = cv2.VideoCapture(file_path)
        frame_num = 0
        user_color = "r"
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = process_frame(frame, debug)

            landmarks_per_frame.append(landmarks)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

            display_pose_3d(landmarks, user_color, ax, zorder=2)

            # with open(TRUTH_PATH, "r") as f:
            #     truth_landmarks = json.load(f)

            # score = find_distance(landmarks, truth_landmarks[frame_num])
            # print(score)

            # if truth and score > .54:
            #     display_pose_3d(truth_landmarks[frame_num], truth_color, ax, zorder=1)
            
            if debug:
                plt.show()

            plt.savefig(f"test_data/extend_arms/frame_{frame_num}.png")

            frame_num += 1

        cap.release()

        # with open("test_data/extend_arms.json", "w") as f:
        #     json.dump(landmarks_per_frame, f)

        return landmarks_per_frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run program")
    parser.add_argument(
        "type", type=str, help="The type of input data", choices=["video", "image"]
    )
    parser.add_argument("file_path", type=str, help="The source file path")
    parser.add_argument("--debug", action="store_true", help="Enable debugging mode", default=False)
    parser.add_argument("--truth", action="store_true", help="Enable truth mode", default=False)

    args = parser.parse_args()
    main(args.type, args.file_path, args.debug, args.truth)


# python main.py type:<video|image> <source_file_path> exercise: <extendArms> debugFlag:<True | False>
