from csv import reader
import random

from naive_bayes import GaussianNBClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# Load csv function to load the data

def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Load the UCI wine data

df = load_csv('wine.data')

# Print the data in exploration

#print(df[0])

# Convert strings to floats

for i in range(len(df)):
    for j in range(len(df[i])):
        df[i][j]=float(df[i][j])

# Check if converted correctly
#print(df[0])
#print(type(df[0][0]))


# Split data function

def split_data(data, weight):
    """
    Random split of a data set into training and test data sets
    
    Parameters:
    -----------
    data: array-like, dataset
    weight: float, percentage of data to be used as training
    
    Returns:
    List of two datasets
    """
    train_length = int(len(data) * weight)
    train = []
    for i in range(train_length):
        idx = random.randrange(len(data))
        train.append(data[idx])
        data.pop(idx)
    return [train, data]

# Split data into train and test

train, test = split_data(df, 0.8)


# Define target and features
# Target is the first column in the wine dataset

X_train = []
y_train = []
X_test = []
y_test = []

for i in range(len(train)):
    y_train.append(train[i][0])
    X_train.append(train[i][1:])
    
for i in range(len(test)):
    y_test.append(test[i][0])
    X_test.append(test[i][1:])


# Instantiate the Naive Bayes classifier model made with numpy

model = GaussianNBClassifier()

# Fit the model

model.fit(X_train, y_train)

# Make predictions

y_pred = model.predict(X_test)

# Print the accuracy of the model

print("NaiveBayesClassifier accuracy using given numpy environment: {0:.3f}".format(model.accuracy(y_test, y_pred)))


# Instantiate the Gaussian Naive Bayes classifier model from Scikit-learn

model = GaussianNB()

# Fit the model

model.fit(X_train, y_train)

# Make predictions

y_pred = model.predict(X_test)

# Print the accuracy of the model

print("Gaussian NaiveBayesClassifier accuracy using Scikit-learn: {0:.3f}".format(accuracy_score(y_test, y_pred)))