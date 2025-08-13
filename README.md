# ASL Alphabet Gesture Recognition System ✋🔤

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-red)

A real-time American Sign Language (ASL) alphabet recognition system using deep learning and computer vision. This project classifies hand gestures into ASL letters (A-Z).

## Dataset

The model is extracted from [ASL Alphabet Dataset from Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet):
But I only choose some part of data for my project:
- Total : 14500 images with 29 classes (A-Z, del, nothing and space). Each class contain 500 images. 
- I split into train, test, and validation set:
    + Train set      (70%) : 10171 images
    + Test set       (15%) : 2181 images
    + Validation set (15%) : 2178 images
- For prediction, I found some images from other dataset (noisier, different light) to test the model's ability to generalize data.

## Preprocess
- Resize input images into size 64x64.
- Convert to grayscale.
- Load in batches of 32 (each time, load 32 images).
- Convert labels (A,B,C,....) into one-hot encode.

## Model
My model using CNN to classify letter from image, which use 11 layers.
- Layer 1   : Normalize pixel range from (0,255) to (0,1) and Resize input image shape.
- Layer 2-7 : Contains 3 pair of Conv2D-Maxpooling to extract the features.
- Layer 8   : Flatten all features into 1-dimension array.
- Layer 9   : Layer Dense to compute number of parameters to connect input layer and hidden layer.
- Layer 10  : Dropout 50% of data => reduce overfitting.
- Layer 11  : Layer Dense to compute number of parameters to connect hidden layer and output layer (number of classes).

## 📊 Evaluation

**Accuracy on training set**: 97.18%  
**Accuracy on testing set**: 96.79%
**Test Loss**: 9.21%

## Strengths
- With still images:
    + This model can predict the letter with high accuracy with still images.
    + With data in the same dataset as training set, the model can predict TRUE with confidence 100%.

## Weaknesses
- With still images:
    + This model is sensitive to noisy, so when using another dataset with different lighting or more noise, it gives wrong results.
- With video:
    + It's same for video, when recognizing people's hand through webcam's video and predicting the letter, it is often wrong because of different conditions.

- This model needs the input to be the same as in training (Ex : same image size and channels). Before predicting, we must preprocess the image in the same way, which is not convenient.
- This dataset is too clean, with little noise, all hands is in the center of the image. This makes the model easy to memorize the data instead of learning features that generalize well.
- The hand is captured from limited angles.
