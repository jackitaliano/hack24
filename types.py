from pydantic import BaseModel
import numpy as np

class Frame(BaseModel):
    frame: np.array
    keypoints: np.array