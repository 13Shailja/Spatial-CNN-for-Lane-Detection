# SCNN (Spatial CNN) for Lane Detection

This repository contains training code and uses a Spatial CNN (SCNN) for lane detection in road images. SCNN is a deep learning model designed for lane detection tasks. It incorporates message-passing between pixels in the image to improve the accuracy of lane predictions.

## Table of Contents
1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Data Preprocessing](#data-preprocessing)
4. [Training](#training)
5. [Model Architecture](#model-architecture)
6. [Results](#results)
7. [References](#references)

## Introduction
This project aims to detect and delineate lane markings on road images. The SCNN model is used to achieve this, and it involves several components, including message-passing layers and a combination of segmentation and existence prediction.

## Setup
Before using this code, you must set up the necessary dependencies. Make sure you have the following libraries installed:

- `matplotlib`: For plotting results.
- `numpy`: For numerical operations.
- `pandas`: For handling data.
- `PIL`: Python Imaging Library for image processing.
- `cv2`: OpenCV for computer vision tasks.
- `scikit-learn`: For preprocessing data and evaluation metrics.
- `torch`: PyTorch deep learning framework.

## Data Preprocessing
The code includes functions for preprocessing lane detection data. It loads images and corresponding label files, extracts lane coordinates, and normalizes the data.

## Training
The training process is defined, including dataset creation, data augmentation, and training using PyTorch's DataLoader. The model architecture, including message-passing layers, is described, and the loss function is defined. The code also establishes the learning rate scheduler and GPU/CPU compatibility.

## Model Architecture
The SCNN model architecture is defined, which includes a VGG16-based backbone network, message-passing layers, and segmentation and existence prediction branches.

## Results
The code provides a framework for training and evaluating the SCNN model for lane detection. You can visualize the results and analyze the performance of the model on your dataset.

## References
- [SCNN: Spatial CNN for Traffic Lane Detection](https://arxiv.org/abs/1712.02190)
- [PyTorch](https://pytorch.org/)

Please note that this README overviews the code and its functionality. Refer to the code comments and associated documentation for detailed usage and implementation.
