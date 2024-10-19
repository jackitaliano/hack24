import argparse
import preprocess.preprocess as preprocess
import depth.depth as depth
import cv2
import matplotlib.pyplot as plt
from pose import pose, draw, process_data


def process_frame(np_img, debug):
    cropped_image = preprocess.inference_image(np_img, debug)
    print("cropped")

    depth_map = depth.inference_image(cropped_image, debug)

    print("depth")
    landmarks_results = pose.get_landmarks(cropped_image, debug)

    relevant_landmarks = process_data.get_relevant_landmarks(
        landmarks_results.pose_landmarks
    )
    processed_landmarks = process_data.get_processed_landmarks(relevant_landmarks)

    relevant_landmarks_results = landmarks_results
    relevant_landmarks_results.pose_landmarks = relevant_landmarks

    print(processed_landmarks)
    # if debug:
    #     print("showing")
    #     annotated_img = draw.draw_landmarks_on_image(
    #         cropped_image, relevant_landmarks_results
    #     )
    #     plt.imshow(annotated_img)
    #     plt.show()


def main(type, file_path, debug):
    if type == "image":
        np_img = plt.imread(file_path)
        process_frame(np_img, debug)

    elif type == "video":
        cap = cv2.VideoCapture(file_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            process_frame(frame, debug)
        cap.release()


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
