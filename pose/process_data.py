import numpy as np
from numpy.ma.core import indices
import numpy.typing as npt

LAND_MARK_NAMES = [
    "nose",
    "neck",
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "pelvis",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

REL_INDICES = np.array([11, 12, 13, 14, 15, 16, 25, 26, 27, 28])

def get_np_landmarks(landmarks):
    landmarks = landmarks[0]
    np_landmarks = np.array(
        [np.array([landmark.x, landmark.y]) for landmark in landmarks]
    )

    return np_landmarks


def get_processed_landmarks(np_landmarks):
    shoulders = np_landmarks[11:13]
    hips = np_landmarks[23:25]
    neck = np.mean(shoulders, axis=0)
    pelvis = np.mean(hips, axis=0)
    right_hip = np_landmarks[24]
    left_hip = np_landmarks[23]
    nose = np.array([np.mean([np_landmarks[0][0], neck[0]]), np_landmarks[0][1]])

    rel_landmarks = np_landmarks[REL_INDICES]
    rel_landmarks = np.insert(rel_landmarks, 0, nose, axis=0)
    rel_landmarks = np.insert(rel_landmarks, 1, neck, axis=0)
    rel_landmarks = np.insert(rel_landmarks, 8, pelvis, axis=0)
    rel_landmarks = np.insert(rel_landmarks, 4, left_hip, axis=0)
    rel_landmarks = np.insert(rel_landmarks, 5, right_hip, axis=0)

    processed_landmarks = {}
    for i in range(len(rel_landmarks)):
        name = LAND_MARK_NAMES[i]
        val = rel_landmarks[i]
        processed_landmarks[name] = val

    return processed_landmarks
