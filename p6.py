import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split

# Activation function: Step function
def activation_function(x):
    return 1 if x >= 0 else 0

# Perceptron class definition
class Perceptron:
    def __init__(self, input_size):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()

    def predict(self, inputs):
        linear_output = np.dot(inputs, self.weights) + self.bias
        return activation_function(linear_output)

    def train(self, X, y, epochs, learning_rate=0.1):
        mse_history = []
        for epoch in range(epochs):
            mse = 0
            correct_predictions = 0
            for inputs, label in zip(X, y):
                prediction = self.predict(inputs)
                error = label - prediction
                mse += error ** 2
                if prediction == label:
                    correct_predictions += 1
                # Weight update rule
                self.weights += learning_rate * error * inputs
                self.bias += learning_rate * error

            mse_history.append(mse / len(X))
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Correct Predictions: {correct_predictions}/{len(X)}")
        return mse_history

# Logic gate datasets
logic_gates = {
    "AND": {
        "X": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        "y": np.array([0, 0, 0, 1]),
    },
    "OR": {
        "X": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        "y": np.array([0, 1, 1, 1]),
    },
    "NAND": {
        "X": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        "y": np.array([1, 1, 1, 0]),
    },
    "NOR": {
        "X": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        "y": np.array([1, 0, 0, 0]),
    },
    "XOR": {
        "X": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        "y": np.array([0, 1, 1, 0]),
    },
    "XNOR": {
        "X": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        "y": np.array([1, 0, 0, 1]),
    },
}

# Train and evaluate for each logic gate
for gate, data in logic_gates.items():
    print(f"\nLogic Gate: {gate}")
    X, y = data["X"], data["y"]

    # Split into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Initialize perceptron
    perceptron = Perceptron(input_size=2)

    # Train the perceptron
    epochs = 50
    mse_history = perceptron.train(X_train, y_train, epochs=epochs, learning_rate=0.1)

    # Evaluate on training data
    y_train_pred = np.array([perceptron.predict(x) for x in X_train])
    train_precision = precision_score(y_train, y_train_pred, zero_division=1)
    train_recall = recall_score(y_train, y_train_pred, zero_division=1)
    train_f1 = f1_score(y_train, y_train_pred, zero_division=1)
    train_accuracy = accuracy_score(y_train, y_train_pred)

    # Evaluate on testing data
    y_test_pred = np.array([perceptron.predict(x) for x in X_test])
    test_precision = precision_score(y_test, y_test_pred, zero_division=1)
    test_recall = recall_score(y_test, y_test_pred, zero_division=1)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=1)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print("Code by Kartikey.S.Rana")
    # Print metrics
    print("Training Metrics:")
    print(f"Precision: {train_precision}")
    print(f"Recall: {train_recall}")
    print(f"F1 Score: {train_f1}")
    print(f"Accuracy: {train_accuracy}")

    print("\nTesting Metrics:")
    print(f"Precision: {test_precision}")
    print(f"Recall: {test_recall}")
    print(f"F1 Score: {test_f1}")
    print(f"Accuracy: {test_accuracy}")

    # Plot MSE history
    plt.figure(figsize=(8, 5))
    plt.plot(range(epochs), mse_history, label=f"{gate} MSE")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error")
    plt.title(f"MSE Over Epochs for {gate}")
    plt.legend()
    plt.grid()
    plt.show()