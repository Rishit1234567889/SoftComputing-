# Practical 1 : Create a perceptron with appropriate number of inputs and outputs. Train it using fixed increment learning algorithm until no change in weights is required. Output the final weights.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
# Step 1: Define the AND gate dataset (inputs and expected outputs)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Inputs (two features)
y = np.array([0, 0, 0, 1])  # Expected outputs for AND gate

# Step 2: Initialize weights and bias
W = np.zeros(X.shape[1])  # Initialize weights to zeros (for 2 inputs, we have 2 weights)
b = 0  # Initialize bias to 0

# Learning rate and maximum epochs
learning_rate = 0.01
epochs = 100  # We'll limit to 100 epochs in case convergence doesn't happen early

# To store MSE for plotting
mse_list = []

# Step 3: Fixed Increment Learning Algorithm
def perceptron_predict(X, W, b):
    """Predict the output of the perceptron using current weights and bias"""
    return np.where(np.dot(X, W) + b >= 0, 1, 0)

# Training loop
for epoch in range(epochs):
    errors = 0
    mse = 0
    for i in range(X.shape[0]):
        # Predict the output
        y_pred = perceptron_predict(X[i], W, b)

        # Update weights and bias only if there's an error
        if y_pred != y[i]:
            W += learning_rate * (y[i] - y_pred) * X[i]  # Fixed increment learning rule
            b += learning_rate * (y[i] - y_pred)
            errors += 1  # Count errors for this epoch

        # MSE calculation
        mse += (y[i] - y_pred) ** 2

    # Calculate and store MSE for this epoch
    mse_list.append(mse / X.shape[0])

    # Stop training if no errors (perfect classification)
    if errors == 0:
        break

# Step 4: Output final weights and bias
print("Final weights:", W)
print("Final bias:", b)

# Step 5: Evaluate the model on Training Data
y_train_pred = perceptron_predict(X, W, b)  # Get final predictions for training data
training_metrics = {
    'Dataset': 'Training',
    'Accuracy': accuracy_score(y, y_train_pred) * 100,
    'Precision': precision_score(y, y_train_pred, average='binary') * 100,
    'Recall': recall_score(y, y_train_pred, average='binary') * 100,
    'F1 Score': f1_score(y, y_train_pred, average='binary') * 100,
}

# Simulate Testing Data (as no separate testing data is defined for AND gate)
X_test = np.array([[1, 1], [0, 0], [1, 0], [0, 1]])
y_test = np.array([1, 0, 0, 0])  # Expected outputs for AND gate
y_test_pred = perceptron_predict(X_test, W, b)

testing_metrics = {
    'Dataset': 'Testing',
    'Accuracy': accuracy_score(y_test, y_test_pred) * 100,
    'Precision': precision_score(y_test, y_test_pred, average='binary') * 100,
    'Recall': recall_score(y_test, y_test_pred, average='binary') * 100,
    'F1 Score': f1_score(y_test, y_test_pred, average='binary') * 100,
}

# Combine metrics
metrics = [training_metrics, testing_metrics]
metrics_df = pd.DataFrame(metrics)

# Display metrics
print("\n--- Evaluation Metrics ---")
print(metrics_df)

# Confusion Matrix and Classification Report
print("\nConfusion Matrix (Training Data):")
print(confusion_matrix(y, y_train_pred))

print("\nClassification Report (Training Data):")
print(classification_report(y, y_train_pred, target_names=['Class 0 (AND False)', 'Class 1 (AND True)']))

print("\nConfusion Matrix (Testing Data):")
print(confusion_matrix(y_test, y_test_pred))

print("\nClassification Report (Testing Data):")
print(classification_report(y_test, y_test_pred, target_names=['Class 0 (AND False)', 'Class 1 (AND True)']))

print("Output by Shourya")

# Step 6: Plot the MSE graph
plt.figure(figsize=(8, 6))
plt.plot(mse_list, marker='o', color='b', label='MSE')
plt.title('MSE per Epoch (AND Gate Perceptron)')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)
plt.show()












