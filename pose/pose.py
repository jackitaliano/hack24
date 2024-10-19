import mediapipe as mp
import matplotlib.pyplot as plt
import numpy.typing as npt

import draw


def get_landmarks(image: npt.NDArray, full: bool = True):
    model_path = "./pose_landmarker_lite.task"

    if full:
        model_path = "./pose_landmarker_full.task"

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


img = "./man.jpg"
img = plt.imread(img)
landmark_results = get_landmarks(img)
annotated = draw.draw_landmarks_on_image(img, landmark_results)

plt.imshow(annotated)
plt.show()
