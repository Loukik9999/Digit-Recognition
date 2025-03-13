# Handwritten Digit Recognition using CNN

## Overview
This project implements a Convolutional Neural Network (CNN) for recognizing handwritten digits using deep learning. The model is trained on the MNIST dataset, which consists of 60,000 training images and 10,000 test images of handwritten digits (0-9).

## Features
- Uses a CNN model for accurate digit recognition
- Implements deep learning techniques for feature extraction
- Trained and tested on the MNIST dataset
- Achieves high accuracy in classification

## Prerequisites
Before running the notebook, ensure you have the following dependencies installed:
- Python 3.x
- Jupyter Notebook
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn

1. Open the Jupyter Notebook:
   ```sh
   jupyter notebook digits_recognition_cnn_hands_on.ipynb
   ```
2. Run all cells to train and evaluate the CNN model.
3. The model will display accuracy metrics and sample predictions.

## Project Structure
- `digits_recognition_cnn_hands_on.ipynb`: Main Jupyter Notebook for training and testing the model.
- `README.md`: Documentation for the project.

## Dataset
The project uses the MNIST dataset, which is publicly available in the `tensorflow.keras.datasets` module.

## Model Architecture
The CNN model consists of:
- Convolutional layers for feature extraction
- Max pooling layers for dimensionality reduction
- Fully connected layers for classification
- Softmax activation for multi-class classification

## Results
The trained model achieves high accuracy on the test dataset and can effectively recognize handwritten digits.

## Acknowledgments
- MNIST Dataset: Yann LeCun et al.
- TensorFlow and Keras for deep learning framework support.







