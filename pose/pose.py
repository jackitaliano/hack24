import mediapipe as mp
import numpy.typing as npt
import numpy as np

LITE_MODEL_PATH = "./pose/pose_landmarker_lite.task"
FULL_MODEL_PATH = "./pose/pose_landmarker_full.task"


def get_landmarks(image: npt.NDArray, debug: bool = False, full: bool = True):
    model_path = LITE_MODEL_PATH

    image = np.array(image)

    if full:
        model_path = FULL_MODEL_PATH

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        pose_landmarker_result = landmarker.detect(mp_image)
        return pose_landmarker_result
