
 # loads the training data into a variable called training_spam.
import numpy as np
from IPython.display import HTML,Javascript, display

training_spam = np.loadtxt(open("data/testing_spam.csv"), delimiter=",").astype(int)
print("Shape of the spam training data set:", training_spam.shape)
print(training_spam)

# loads the testing_data but as spam
testing_spam = np.loadtxt(open("data/training_spam.csv"), delimiter=",").astype(int)
print("Shape of the spam testing data set:", testing_spam.shape)
print(testing_spam)


# Cleans data
def preprocessing(training_spam):
    # Remove any rows with all zeros
    training_spam = training_spam[~np.all(training_spam == 0, axis=1)]
    
    # Remove any duplicate rows
    training_spam = np.unique(training_spam, axis=0)
    
    return training_spam
training_spam = preprocessing(training_spam)

print("\nShape of the spam training data set after preprocessing:", training_spam.shape)
print("After preprocessing:")
print(training_spam)

#  Naive Bayes Classifier Computation and Logic

class NaiveBayesClassifier:
    def __init__(self):
        self.log_class_priors = None
        self.log_class_conditional_likelihoods = None

    def train(self, data, alpha=1.0):
        # Extract the binary response variable from the first column
        response_variable = data[:, 0].astype(int)  # Convert to integers

        # Extract the feature matrix from the remaining columns
        features = data[:, 1:]

        # Get the number of features
        n_features = features.shape[1]

        # Initialize arrays to store counts
        class_counts = np.zeros((2, n_features))  # Counts of each feature for each class
        class_totals = np.zeros(2)  # Total counts for each class

        # Iterate over each sample
        for i, c in enumerate(response_variable):
            # Increment class total count
            class_totals[int(c)] += 1  # Convert c to integer
            # Increment counts of each feature for the current class
            class_counts[int(c)] += features[i]

        # Apply Laplace smoothing if alpha > 0
        if alpha > 0:
            class_counts += alpha
            class_totals += alpha * 2  # 2 as there are two classes

        # Calculate class-conditional likelihoods
        self.log_class_conditional_likelihoods = np.log(class_counts / class_totals[:, np.newaxis])

        # Calculate class priors
        self.log_class_priors = np.log(class_totals / np.sum(class_totals))

    def predict(self, new_data):
        # Calculate the logarithm of the posterior probability for each class
        posterior_log_probs = np.dot(new_data, self.log_class_conditional_likelihoods.T) + self.log_class_priors

        # Predict the class with the highest posterior probability for each instance
        class_predictions = np.argmax(posterior_log_probs, axis=1)

        return class_predictions

# This skeleton code simply classifies every input as ham

class SpamClassifier:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classifier = NaiveBayesClassifier()

    def train(self, data):
        self.classifier.train(data, alpha=self.alpha)

    def predict(self, new_data):
        return self.classifier.predict(new_data)

def create_classifier(training_data):
    classifier = SpamClassifier(alpha=1)
    classifier.train(training_data)
    return classifier
    
    classifier = create_classifier(training_spam)

