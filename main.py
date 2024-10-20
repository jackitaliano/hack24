import argparse
import preprocess.preprocess as preprocess
import depth.depth as depth
import cv2
import matplotlib.pyplot as plt
from pose import pose, process_data, draw
from matplotlib.patches import Circle


def normalize_landmarks(landmarks):
    pelvis = landmarks["pelvis"]

    for keypoint in landmarks:
        landmarks[keypoint][0] -= pelvis[0]
        landmarks[keypoint][1] -= pelvis[1]
        landmarks[keypoint][2] -= pelvis[2]

    return landmarks


def process_frame(cropped_image, debug):
    # print("cropping image")
    # cropped_image = preprocess.inference_image(np_img, debug)

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

    # print("getting depth")
    # landmarks = depth.inference_image(cropped_image, debug, processed_landmarks)

    landmarks = normalize_landmarks(processed_landmarks)

    return landmarks


def main(type, file_path, debug):
    if type == "image":
        np_img = plt.imread(file_path)
        landmarks = process_frame(np_img, debug)
        print(landmarks)
        return landmarks

    elif type == "video":
        landmarks_per_frame = []

        cap = cv2.VideoCapture(file_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = process_frame(frame, debug)
            landmarks_per_frame.append(landmarks)
        cap.release()

        print(landmarks_per_frame)
        return landmarks_per_frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run program")
    parser.add_argument(
        "type", type=str, help="The type of input data", choices=["video", "image"]
    )
    parser.add_argument("file_path", type=str, help="The source file path")
    parser.add_argument("debug", type=bool, help="Debugging mode", default=False)

    args = parser.parse_args()
    main(args.type, args.file_path, args.debug)


# python main.py type:<video|image> <source_file_path> debugFlag:<True | False>
