import cv2
import numpy as np
import matplotlib.pyplot as plt
import depth.depth as depth
from PIL import Image
from transformers import pipeline
import preprocess.preprocess as preprocess
from pose import pose, draw, process_data


def get_depth(frame):
    image = Image.fromarray(frame)
    pipe = pipeline(
        task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf"
    )
    depth = pipe(image)["depth"]

    return np.array(depth)


def get_pose(frame):
    print("getting pose landmarks")
    dimensions = frame.shape

    landmarks_results = pose.get_landmarks(frame)
    np_landmarks = process_data.get_np_landmarks(landmarks_results.pose_landmarks)
    processed_landmarks = process_data.get_processed_landmarks(np_landmarks)

    annotated_image = draw.draw_landmarks_on_image(
        frame, processed_landmarks, dimensions
    )

    return annotated_image


def main():
    FILE_PATH = "test_data/walk.mp4"
    cap = cv2.VideoCapture(FILE_PATH)

    pose_video = []
    depth_video = []

    i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print("Frame,", i)

        depth_frame = get_depth(frame)
        depth_video.append(depth_frame)

        i += 1

    cap.release()

    height, width = depth_video[0].shape[:2]
    print(height, width)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4
    fps = 30  # Frames per second
    output_file = "output_video.mp4"

    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for image in depth_video:
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video_writer.write(bgr_image)

    video_writer.release()


if __name__ == "__main__":
    main()
