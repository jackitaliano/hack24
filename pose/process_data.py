import numpy as np
import numpy.typing as npt

LAND_MARK_NAMES = [
    "left_eye",
    "right_eye",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]

# d = {
#     "left_eye": [1, 2],
#     "right_eye": [2, 3]
# }

INDICES = np.array([2, 5] + list(range(11, 32)))


def get_relevant_landmarks(landmarks):
    landmarks = landmarks[0]

    return [[landmarks[i] for i in INDICES]]


def get_processed_landmarks(landmarks):
    landmarks = landmarks[0]

    processed_landmarks = {}
    for i in range(len(landmarks)):
        landmark = landmarks[i]

        y = landmark.y
        x = landmark.x

        name = LAND_MARK_NAMES[i]
        value = (x, y)

        processed_landmarks[name] = value

    return processed_landmarks
