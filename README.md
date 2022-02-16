# Drivers' Fatigue Detection System

## 1. Introduction

The program is an algorithm for feature point regression of human faces to detect yawning and eye-closing behavior in driving for fatigue detection and alerting. The algorithm uses ResNet-50 with dilated convolution as the backbone trained on Wider Facial Landmark in-the-wild dataset. The algorithm introduced a different way of calculating Mean Square Loss by introducing weights calculated by the Euler angle and attributes of the training samples. After the feature points are inferred, the algorithm asserts fatigue by calculating the distance between eye lids and lips: if the distance between eye lids are smaller than a threshold for three seconds and if the distance between upper and lower lips is larger than a certain threshold for five seconds. The result is tested on YawDD Dataset.

## 2. Prerequisites

Python 3.7

PyTorch==1.7.1

Windows10 20H2

## 3. Requirements

Run `pip install -r requirements.txt`

## 4. Training

First run data-preprocessing.py, it will generate folder Img Dataset with sub-folders： imgs and annotations, which has training and testing set. Then run training.py which will generate a folder models which saves trained model in ckpt.pth.tar.

## 5. Testing

File yawning_detection.py has two segments of code, which both include testing and yawning detection. Download the YawDD dataset and put in the video file path in the parameter video_file to test the result.

## 6. Details on new loss functions.

In order to improve the model's performance on the special samples, those with occlusion, exaggerated expression, and tilted face position. The loss function is revised to include two more weights by calculating Euler angle and the attributes in the WFLW dataset annotations. The new loss function is as the following:
$$ \frac{1}{M}\sum_{m=1}^{M}\sum_{n=1}^{N}\Big(\sum_{c=1}^{C}w_n^c\sum_{k=1}^K(1-cos\theta_n^k) \Big)\parallel d_n^m\parallel_2^2$$                                           
where K represents three axes of Euler angle, pitch, yaw, roll, C represents 6 types of attributes in the WFLW dataset, $w_c$ represents the weight determined by the reciprocal of the fraction of samples with attribute c in the total dataset.
