import cv2
import numpy as np


class EulerAngleCalc():
    def __init__(self, camw=256, camh=256):
        self.landmarks_3D = np.float32([
            [6.825897, 6.760612, 4.402142],  # LEFT_EYEBROW_LEFT,
            [1.330353, 7.122144, 6.903745],  # LEFT_EYEBROW_RIGHT,
            [-1.330353, 7.122144, 6.903745],  # RIGHT_EYEBROW_LEFT,
            [-6.825897, 6.760612, 4.402142],  # RIGHT_EYEBROW_RIGHT,
            [5.311432, 5.485328, 3.987654],  # LEFT_EYE_LEFT,
            [1.789930, 5.393625, 4.413414],  # LEFT_EYE_RIGHT,
            [-1.789930, 5.393625, 4.413414],  # RIGHT_EYE_LEFT,
            [-5.311432, 5.485328, 3.987654],  # RIGHT_EYE_RIGHT,
            [-2.005628, 1.409845, 6.165652],  # NOSE_LEFT,
            [-2.005628, 1.409845, 6.165652],  # NOSE_RIGHT,
            [2.774015, -2.080775, 5.048531],  # MOUTH_LEFT,
            [-2.774015, -2.080775, 5.048531],  # MOUTH_RIGHT,
            [0.000000, -3.116408, 6.097667],  # LOWER_LIP,
            [0.000000, -7.415691, 4.070434],  # CHIN
        ])
        c_x = camw / 2
        c_y = camh / 2
        f_x = c_x / np.tan(60 / 2 * np.pi / 180)
        f_y = f_x
        self.camera_matrix = np.float32([[f_x, 0.0, c_x], [0.0, f_y, c_y],
                                    [0.0, 0.0, 1.0]])
        self.camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])
        self.track_points = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]

    def calculate(self, landmarks_2D):
        if type(landmarks_2D) is not np.ndarray:
            landmarks_2D = np.asarray(landmarks_2D)
        _, rvec, tvec = cv2.solvePnP(self.landmarks_3D, landmarks_2D, self.camera_matrix, self.camera_distortion)
        rmat, _ = cv2.Rodrigues(rvec)
        pose_mat = cv2.hconcat((rmat, tvec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
        return map(lambda k: k[0], euler_angles)
