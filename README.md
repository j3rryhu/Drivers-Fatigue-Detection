# Drivers' Fatigue Detection System

## 1. Introduction

The program is an algorithm for feature point regression of human faces to detect yawning and eye-closing behavior in driving for fatigue detection and alerting. The algorithm uses ResNet-50 as the backbone trained on Wider Facial Landmark in-the-wild dataset. The algorithm introduced a different way of calculating Mean Square Loss by introducing weights calculated by the Euler angle and attributes of the training samples. After the feature points are inferred, the algorithm asserts fatigue by calculating the distance between eye lids and lips: if the distance between eye lids are smaller than a threshold for three seconds and if the distance between upper and lower lips is larger than a certain threshold for five seconds. The result is tested on YawDD Dataset.

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

