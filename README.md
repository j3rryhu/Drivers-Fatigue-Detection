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

## 6. New Loss Function

The new loss function adds weights calculated by Euler angle and the attribute of images to increase the loss on the irregular samples, for example, those with exaggerated expression or faces occluded. It could extensively improve the performance of the mode. The new function is as following:

$$\frac{1}{M}\sum^{M}_{m=1}\sum^{N}_{n=1}(\sum^{C}_{c=1}w^c_n\sum^{K}_{k=1}(1-cos\theta^k_n))\parallel d^m_n \parallel^2_2$$

where $w^c_n$ represents the weight of attribute of a sample determined by the reciprocal of the fraction of the number of one attribute in a set over the total number of samples in the dataset. $\theta^k_n$ is the Euler Angle for each sample, n, where k represents three angles of Euler Angle, pitch, yaw, roll.

## 7. Visualize annotations

Run test_annotation.py to visualize the landmark on image
