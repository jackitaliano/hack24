from transformers import pipeline
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def inference_image(frame, debug):
    image = Image.fromarray(frame)
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    depth = pipe(image)["depth"]
    
    if debug:    
        plt.imshow(depth, cmap="jet")
        plt.show()

    return depth