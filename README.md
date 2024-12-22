# deep-learning-flower-classifier
# Pre-trained Image Classifier for Dog Breeds and Flower Classes
This repository contains a project that uses a pre-trained deep learning model to classify images into specific categories. The project is split into two main parts: a development notebook and a command-line application.

# Project Overview
Part 1: Development Notebook
A Jupyter Notebook demonstrates the implementation of data preprocessing, model training, and evaluation using PyTorch and torchvision.
Part 2: Command-Line Application
Scripts (train.py and predict.py) allow users to train a model and make predictions via a CLI.
Features
Part 1: Development Notebook
Data Preprocessing:

Augmentation using torchvision.transforms (random scaling, rotations, mirroring, cropping).
Normalization for training, validation, and testing datasets.
Datasets loaded with torchvision.datasets.ImageFolder.
Model Setup:

Pre-trained networks (e.g., VGG16) loaded with frozen parameters.
Custom feedforward classifier defined and trained.
Training and Evaluation:

Validation loss and accuracy displayed during training.
Final model accuracy evaluated on test data.
Model checkpoint saved with hyperparameters and class_to_idx dictionary.
Utility Functions:

process_image: Converts a PIL image into input for the trained model.
predict: Returns the top K probable classes for an input image.
Sanity check: Displays an image and its top 5 predicted classes using Matplotlib.
Part 2: Command-Line Application
Training (train.py):

Train a new model on a dataset of images.
Log training and validation losses as well as validation accuracy.
Supports multiple architectures (e.g., VGG16, ResNet).
Hyperparameter tuning (learning rate, hidden units, epochs).
Optional GPU training.
Prediction (predict.py):

Classify an image using a saved model checkpoint.
Display top K predicted classes with probabilities.
Option to load category names from a JSON file.
Supports GPU for prediction.
Files Submitted
Development Notebook: Image_Classifier.ipynb
Training Script: train.py
Prediction Script: predict.py
Model Checkpoint: checkpoint.pth
Category Mapping: cat_to_name.json

# Requirements
Python 3.x
PyTorch
torchvision
Matplotlib
NumPy
PIL
# How to Use Training bash Copy code
python train.py --data_dir <path_to_data> --arch vgg16 --learning_rate 0.001 --hidden_units 512 --epochs 5 --gpu  

Predicting bash Copy code
# python predict.py <path_to_image> <checkpoint.pth> --top_k 5 --category_names cat_to_name.json --gpu  

# Future Improvements
Add support for additional architectures.
Implement web or GUI-based prediction interface.
