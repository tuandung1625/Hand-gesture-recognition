# ASL Alphabet Gesture Recognition System ✋🔤

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-red)

A real-time American Sign Language (ASL) alphabet recognition system using deep learning and computer vision. This project classifies hand gestures from webcam input into ASL letters (A-Z).

## Features

- Real-time hand detection and classification
- Pre-trained CNN model for ASL alphabet (A-Z)
- Dataset preprocessing and splitting utilities
- Webcam interface with visual feedback

## Dataset

The model was trained on the [ASL Alphabet Dataset from Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) containing:
- 87,000 images of ASL gestures
- 29 classes (A-Z plus space, delete, and nothing)
- 200x200px RGB images
NOTE: my model represent the performance on 4 letters A, B, C, D just for research and study
- Each letter dataset have about 500 images of ASL gestures in different lighting and angles
- The data then split into 80% for training and 20% for testing

## 📊 Evaluation Metrics

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **A** | 0.81      | 1.00   | 0.90     | 110     |
| **B** | 1.00      | 0.99   | 1.00     | 109     |
| **C** | 1.00      | 0.77   | 0.87     | 116     |
| **D** | 0.93      | 0.95   | 0.94     | 111     |

**Overall Accuracy**: 93%  
**Test Loss**: 0.624
