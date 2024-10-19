from transformers import pipeline
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def inference_image(frame, debug, landmarks):
    image = Image.fromarray(frame)
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    depth = pipe(image)["depth"]
    
    depth = np.array(depth)
    depth = depth.astype(np.float64) / 255.0
    depth -= .5
    depth *- 2.0

    if debug:    
        plt.imshow(depth, cmap="jet")
        plt.show()

    for keypoint in landmarks:
        landmarks[keypoint] = [landmarks[keypoint][0], landmarks[keypoint][1], depth(landmarks[keypoint][1], landmarks[keypoint][0])]        
    
    return landmarks