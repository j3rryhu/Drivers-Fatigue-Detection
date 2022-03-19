import cv2
import numpy as np
import random


class DataAug:
    def __init__(self,
                 saturation_var=0.5,
                 brightness_var=0.5,
                 lighting_std=0.5,
                 horizontal_flip_probability=0.5,
                 vertical_flip_probability=0.5,
                 do_random_crop=False,
                 zoom_range=[0.75, 1.25],
                 translation_factor=.3):

        self.color_jitter = []
        self.saturation_var = saturation_var
        if brightness_var:
            self.brightness_var = brightness_var
            self.color_jitter.append(self.brightness)
        self.lighting_std = lighting_std
        self.horizontal_flip_probability = horizontal_flip_probability
        self.vertical_flip_probability = vertical_flip_probability
        self.do_random_crop = do_random_crop
        self.zoom_range = zoom_range
        self.translation_factor = translation_factor

    def _do_random_crop(self, image, bbox, landmark):
        h, w, c = image.shape
        resize_factor = 1.5
        face_width = bbox[2]-bbox[0]
        face_height = bbox[3]-bbox[1]
        center_x = int(bbox[2]+bbox[0])//2
        center_y = int(bbox[3]+bbox[1])//2
        x_offset = random.randint(-int(self.translation_factor * face_width), int(self.translation_factor * face_width))
        y_offset = random.randint(-int(self.translation_factor * face_height), int(self.translation_factor * face_height))
        new_center_x = center_x+x_offset
        new_center_y = center_y+y_offset
        upperleftx = max(new_center_x-face_width//2, 0)
        upperlefty = max(new_center_y-face_height//2, 0)
        for idx in range(0, len(bbox), 2):
            bbox[idx] = max(bbox[idx]-upperleftx, 0)
            bbox[idx+1] = max(bbox[idx]-upperlefty, 0)
        for idx in range(0, len(landmark), 2):
            landmark[idx] = max(landmark[idx]-upperleftx, 0)
            landmark[idx+1] = max(landmark[idx+1]-upperlefty, 0)
        image = image[upperlefty:min(upperlefty+face_height, h), upperleftx:min(upperleftx+face_width, w)]
        image = cv2.resize(image, (int(w*resize_factor), int(h*resize_factor)))
        for i in range(0, len(bbox), 2):
            bbox[i]*=resize_factor
            bbox[i+1]*=resize_factor
        for i in range(0, len(landmark), 2):
            landmark[i]*=resize_factor
            landmark[i+1]*=resize_factor
        return image, bbox, landmark

    def brightness(self, image_array):
        alpha = 2 * np.random.random() * self.brightness_var
        alpha = alpha + 1 - self.saturation_var
        image_array = alpha * image_array
        return np.clip(image_array, 0, 255)

    def lighting(self, image_array):
        covariance_matrix = np.cov(image_array.reshape(-1, 3) /
                                   255.0, rowvar=False)
        eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)
        noise = np.random.randn(3) * self.lighting_std
        noise = eigen_vectors.dot(eigen_values * noise) * 255
        image_array[0] = image_array[0] + noise[0]
        image_array[1] = image_array[1] + noise[1]
        image_array[2] = image_array[2] + noise[2]
        return np.clip(image_array, 0, 255)

    def horizontal_flip(self, image, bbox, landmark):
        h, w, c = image.shape
        dst = np.zeros((h, w, c), np.uint8)
        for d in range(c):
            for i in range(w):
                for j in range(h):
                    dst[j][i][d] = image[j][w-1-i][d]
        bbox[0] = w-1-bbox[0]
        bbox[2] = w-1-bbox[2]
        for idx in range(0, len(landmark), 2):
            landmark[idx] = w-1-landmark[idx]
        return dst, bbox, landmark

    def vertical_flip(self, image, bbox, landmark):
        h, w, c = image.shape
        dst = np.zeros((h, w, c), np.uint8)
        for d in range(c):
            for i in range(w):
                for j in range(h):
                    dst[j][i][d] = image[h-j-1][i][d]
        bbox[1] = h - 1 - bbox[1]
        bbox[3] = h - 1 - bbox[3]
        for idx in range(0, len(landmark), 2):
            landmark[idx+1] = h - 1 - landmark[idx+1]
        return dst, bbox, landmark
