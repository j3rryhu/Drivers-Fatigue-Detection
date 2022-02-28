import cv2
import numpy as np
import scipy.ndimage as ndi
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
        width = image.shape[1]
        height = image.shape[2]
        center_x = width//2
        center_y = height//2
        x_offset = random.randint(-self.translation_factor * width, self.translation_factor * width)
        y_offset = random.randint(-self.translation_factor * height, self.translation_factor * height)
        new_center_x = center_x+x_offset
        new_center_y = center_y+y_offset
        upperleftx = new_center_x+x_offset
        upperlefty = new_center_y+y_offset
        for idx in range(0, len(bbox), 2):
            bbox[idx] = min(bbox[idx]-upperleftx, 0)
            bbox[idx+1] = min(bbox[idx]-upperlefty, 0)
        for idx in range(0, len(landmark), 2):
            landmark[idx] = min(landmark[idx]-upperleftx, 0)
            landmark[idx+1] = min(landmark[idx+1]-upperlefty, 0)
        image = image[upperlefty:height, upperleftx:width]
        image = cv2.resize(image, (width, height))
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
        image_array = image_array + noise
        return np.clip(image_array, 0, 255)

    def horizontal_flip(self, image, bbox, landmark):
        c, w, h = image.shape
        dst = np.zeros((c, w, h), np.uint8)
        for d in range(c):
            for i in range(w):
                for j in range(h):
                    dst[d][i][j] = image[d, w-1-i, h]
        bbox[0] = w-1-bbox[0]
        bbox[2] = w-1-bbox[2]
        for idx in range(0, len(landmark), 2):
            landmark[idx] = w-1-landmark[idx]
        return dst, bbox, landmark

    def vertical_flip(self, image, bbox, landmark):
        c, w, h = image.shape
        dst = np.zeros((c, w, h), np.uint8)
        for d in range(c):
            for i in range(w):
                for j in range(h):
                    dst[d][i][j] = image[d, w, h-j-1]
        bbox[1] = h - 1 - bbox[1]
        bbox[3] = h - 1 - bbox[3]
        for idx in range(0, len(landmark), 2):
            landmark[idx+1] = h - 1 - landmark[idx+1]
        return image, bbox, landmark

    def preprocess_images(self, dataset):
        augment_proportion = random.uniform(0.2, 0.4)
        augment_num = int(len(dataset.samples)*augment_proportion)
        count = 0
        rand_idices = []
        while count<augment_num:
            rand_idx = random.randint(0, len(dataset.samples))
            if rand_idx not in rand_idices:
                rand_idices.append(rand_idx)
            else:
                continue
            img, lm, boundingbox, attribute, ea = dataset.samples[rand_idx]
            # RANDOM CROP
            cropped_img, cropped_bbox, cropped_lm = self._do_random_crop(img, boundingbox, lm)
            rand_insert = random.randint(0, len(dataset.samples))
            attribute[4] = 1
            dataset.samples.insert(rand_insert, [cropped_img, cropped_lm, cropped_bbox, attribute, ea])
            # FLIP
            flip_method = random.randint(0, 1)
            if flip_method:
                flipped_img, flipped_bbox, flipped_lm = self.vertical_flip(img, boundingbox, lm)
            else:
                flipped_img, flipped_bbox, flipped_lm = self.horizontal_flip(img, boundingbox, lm)
            rand_insert = random.randint(0, len(dataset.samples))
            dataset.samples.insert(rand_insert, [flipped_img, flipped_lm, flipped_bbox, attribute, ea])
            # LIGHT
            lit_img = self.lighting(img)
            rand_insert = random.randint(0, len(dataset.samples))
            dataset.samples.insert(rand_insert, [lit_img, lm, boundingbox, attribute, ea])
            # BRIGHTEN
            brightened_img = self.brightness(img)
            rand_insert = random.randint(0, len(dataset.samples))
            dataset.samples.insert(rand_insert, [brightened_img, lm, boundingbox, attribute, ea])
        return dataset.samples


