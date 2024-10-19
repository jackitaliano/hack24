import argparse
import preprocess.preprocess as preprocess
import depth.depth as depth

def main(file_path, debug):
    print(f"Yolov11 on Image: {file_path}")
    cropped_image = preprocess.inference_image(file_path, debug)

    print(f"Depth Estimation on Image: {file_path}")
    depth_map = depth.inference_image(cropped_image, debug)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run program")
    parser.add_argument('file_path', type=str, help='The source file path')
    parser.add_argument("debug", type=bool, help="Debugging mode", default=False)

    args = parser.parse_args()
    main(args.file_path, args.debug)

    
# python main.py <source_file_path> <True | False>