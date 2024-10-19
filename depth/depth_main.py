from transformers import pipeline
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def main():
    frame = "../data/test.png"
    inference(frame)

def inference(frame):
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    image = Image.open(frame)
    depth = pipe(image)["depth"]

    print(np.array(depth))
    plt.imshow(depth, cmap="jet")
    plt.show()

if __name__ == '__main__':
    main()