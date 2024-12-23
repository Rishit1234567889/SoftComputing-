# Practical 9A : Write a program to implement Hebb's rule.
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Step 1: Load the Iris dataset
iris = datasets.load_iris()
# Use only two classes for binary classification (Setosa and Versicolor)
X = iris.data[:100, [0, 2]]  # We select only two features: sepal length and petal length
y = iris.target[:100]  # Select only the first 100 samples (Setosa and Versicolor)

# Convert labels: Setosa (0), Versicolor (1)
y = np.where(y == 0, -1, 1)  # For Hebbian learning, we need to have labels as -1 and 1
print(X[:5], y[:5])

# Step 2: Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Standardize the features (zero mean, unit variance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Learning rate for Hebbian learning
lr = 0.03

# Step 3: Initialize weights to zeros
W = np.zeros(X_train.shape[1])  # We have two features, so two weights

# Step 4: Hebb rule learning (Training)
# To store MSE values for each epoch
mse_train_list = []
mse_test_list = []

# Train for a fixed number of epochs
epochs = 100
for epoch in range(epochs):
    y_train_pred = []
    for i in range(X_train.shape[0]):
        # Update the weights based on Hebbian rule
        W += lr * X_train[i] * y_train[i]

    # Get predictions on training data
    y_train_pred = np.sign(np.dot(X_train, W))
    y_test_pred = np.sign(np.dot(X_test, W))

    # Calculate MSE for both training and testing sets
    mse_train = np.mean((y_train - y_train_pred) ** 2)
    mse_test = np.mean((y_test - y_test_pred) ** 2)

    mse_train_list.append(mse_train)
    mse_test_list.append(mse_test)

# Step 5: Testing the trained model
def predict(X, W):
    """Predict the class using the learned weights"""
    return np.sign(np.dot(X, W))

# Get final predictions for training and testing datasets
y_train_pred = predict(X_train, W)
y_test_pred = predict(X_test, W)

# Step 6: Evaluation metrics for training and testing sets
def evaluate_model(y_true, y_pred, dataset_name):
    print(f"\nEvaluation on {dataset_name} dataset:")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=['Setosa (-1)', 'Versicolor (1)']))

# Evaluate on training data
evaluate_model(y_train, y_train_pred, "Training")

# Evaluate on testing data
evaluate_model(y_test, y_test_pred, "Testing")

# Step 7: Plot the MSE graph for both training and testing sets
plt.figure(figsize=(10, 6))
plt.plot(mse_train_list, label='Train MSE', color='blue')
plt.plot(mse_test_list, label='Test MSE', color='red', linestyle='--')
plt.title('MSE vs Epochs for Train and Test Sets')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend()
plt.grid(True)
plt.show()



# Practical 9B : Write a program to implement Delta rule.
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Step 1: Load the Iris dataset
iris = datasets.load_iris()
# Use only two classes for binary classification (Setosa and Versicolor)
X = iris.data[:100, [0, 2]]  # We select only two features: sepal length and petal length
y = iris.target[:100]  # Select only the first 100 samples (Setosa and Versicolor)

# Convert labels: Setosa (0), Versicolor (1)
y = np.where(y == 0, -1, 1)  # For Delta rule, we need to have labels as -1 and 1

# Step 2: Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Standardize the features (zero mean, unit variance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 3: Initialize weights and bias
W = np.zeros(X_train.shape[1])  # Weights (2 features, so 2 weights)
b = 0  # Bias term

# Learning rate for the Delta rule
lr = 0.01

# Step 4: Train the model using the Delta rule
epochs = 100
mse_train_list = []
mse_test_list = []

for epoch in range(epochs):
    for i in range(X_train.shape[0]):
        # Calculate the prediction (linear combination)
        y_pred = np.dot(X_train[i], W) + b

        # Update rule for the weights and bias (Delta rule)
        W += lr * (y_train[i] - y_pred) * X_train[i]
        b += lr * (y_train[i] - y_pred)

    # After weight updates, calculate the predictions for both training and test sets
    y_train_pred = np.dot(X_train, W) + b
    y_test_pred = np.dot(X_test, W) + b

    # Apply the sign function to convert predictions to -1 or 1
    y_train_pred = np.sign(y_train_pred)
    y_test_pred = np.sign(y_test_pred)

    # Calculate MSE for the current epoch
    mse_train = np.mean((y_train - y_train_pred) ** 2)
    mse_test = np.mean((y_test - y_test_pred) ** 2)

    mse_train_list.append(mse_train)
    mse_test_list.append(mse_test)

# Step 5: Evaluate the model
def evaluate_model(y_true, y_pred, dataset_name):
    print(f"\nEvaluation on {dataset_name} dataset:")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=['Setosa (-1)', 'Versicolor (1)']))

# Final predictions after training
y_train_pred = np.sign(np.dot(X_train, W) + b)
y_test_pred = np.sign(np.dot(X_test, W) + b)

# Evaluate on training data
evaluate_model(y_train, y_train_pred, "Training")

# Evaluate on testing data
evaluate_model(y_test, y_test_pred, "Testing")

# Step 6: Plot the MSE graph for both training and testing sets
plt.figure(figsize=(10, 6))
plt.plot(mse_train_list, label='Train MSE', color='blue')
plt.plot(mse_test_list, label='Test MSE', color='red', linestyle='--')
plt.title('MSE vs Epochs for Train and Test Sets (Delta Rule)')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend()
plt.grid(True)
plt.show()
