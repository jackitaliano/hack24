import argparse
import preprocess.preprocess as preprocess
import depth.depth as depth
import cv2

def process_frame(file_path, debug):
    print(f"Yolov11 on Image: {file_path}")
    cropped_image = preprocess.inference_image(file_path, debug)

    print(f"Depth Estimation on Image: {file_path}")
    depth_map = depth.inference_image(cropped_image, debug)

def main(type, file_path, debug):
    if type == "image":
        process_frame(file_path, debug)
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
    parser.add_argument('type', type=str, help='The type of input data', choices=['video', 'image'])
    parser.add_argument('file_path', type=str, help='The source file path')
    parser.add_argument("debug", type=bool, help="Debugging mode", default=False)

    args = parser.parse_args()
    main(args.type, args.file_path, args.debug)

    
# python main.py type:<video|image> <source_file_path> debugFlag:<True | False>