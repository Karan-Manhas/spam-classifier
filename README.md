# Multinomial Naive Bayes Spam Classifier

This repository contains a Python implementation of a multinomial Naive Bayes classifier for spam detection. The classifier is trained on labeled data and then used to predict whether new messages are spam or not.

## Features of Code

- Loads training and testing data from CSV files.
- Preprocesses the training data by removing rows with all zeros and duplicates.
- Implements a multinomial Naive Bayes classifier to classify messages as spam or not spam.
- Allows setting the Laplace smoothing parameter (`alpha`) for the classifier.
- Provides a wrapper class for the classifier, making it easy to train and predict.
- Includes a sample script to demonstrate the usage of the classifier.

## Requirements

- Jupyter Notebook
- Anaconda Navigation (optional but recommended)

## Usage

1.. Prepare your training and testing data:
   
   - The training data should be in CSV format with each row representing a message and columns representing features (e.g., word counts).
   - The testing data should be similarly formatted.

4. Update the paths to your training and testing data in the provided code (`spam_classifier.py`).

5. Run the provided script in Jupyter Notebooks

## Notes

- Ensure that your training data is labeled correctly. The first column should contain binary labels indicating whether each message is spam or not spam.
- Experiment with different values of the Laplace smoothing parameter (`alpha`) to optimize classifier performance.
- This implementation assumes that the features represent counts of occurrences (e.g., word counts) and uses a Multinomial Naive Bayes model.
