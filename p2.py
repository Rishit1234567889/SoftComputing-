import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/content/ADANIPORTS.csv'  # Replace this with your dataset path
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"File not found at {file_path}. Please check the path.")
    exit()

# Check if required columns exist
required_columns = ['Open', 'High', 'Low', 'Volume', 'Close']
if not all(col in df.columns for col in required_columns):
    print(f"Missing one or more required columns: {required_columns}")
    exit()

# Select relevant columns: we use 'Open', 'High', 'Low', 'Volume' to predict 'Close'
features = df[['Open', 'High', 'Low', 'Volume']]
target = df[['Close']]

# Convert to numpy arrays
X = features.values
y = target.values

# Normalize the data (scale values between 0 and 1)
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Add noise to the testing dataset
X_test_noisy = X_test + np.random.normal(0, 0.05, X_test.shape)  # Adding Gaussian noise

# Define the Neural Network class
class StockPricePredictor:
    def __init__(self, input_size, hidden_size, output_size):
        """Initialize weights and biases for a 1-hidden-layer neural network"""
        print("Initializing StockPricePredictor...")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size) * 0.1
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size) * 0.1
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """Derivative of sigmoid for backpropagation"""
        return x * (1 - x)

    def forward_pass(self, X):
        """Perform a forward pass through the network"""
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.output = self.sigmoid(self.output_layer_input)
        return self.output

    def backward_pass(self, X, y, learning_rate):
        """Update weights and biases using backpropagation"""
        # Calculate errors
        output_error = y - self.output
        output_delta = output_error * self.sigmoid_derivative(self.output)

        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_layer_output)

        # Update weights and biases
        self.weights_hidden_output += np.dot(self.hidden_layer_output.T, output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += np.dot(X.T, hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        """Train the model using the training data"""
        loss_history = []
        for epoch in range(epochs):
            # Forward pass
            self.forward_pass(X)

            # Backward pass
            self.backward_pass(X, y, learning_rate)

            # Calculate loss (Mean Squared Error)
            loss = np.mean(np.square(y - self.output))
            loss_history.append(loss)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
        return loss_history

# Initialize the neural network
input_size = X_train.shape[1]
hidden_size = 8
output_size = 1
learning_rate = 0.01
epochs = 1000

# Train the model
model = StockPricePredictor(input_size, hidden_size, output_size)
loss_history = model.train(X_train, y_train, epochs, learning_rate)

# Plot the loss over epochs
plt.plot(loss_history, label="Loss")
plt.title('Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Predictions on Training Data
predicted_train = model.forward_pass(X_train)

# Predictions on Testing Data (Noisy)
predicted_test = model.forward_pass(X_test_noisy)

# Inverse transform the predictions and actual values
predicted_train_prices = scaler_y.inverse_transform(predicted_train)
actual_train_prices = scaler_y.inverse_transform(y_train)

predicted_test_prices = scaler_y.inverse_transform(predicted_test)
actual_test_prices = scaler_y.inverse_transform(y_test)

# Evaluation Metrics
def calculate_metrics(actual, predicted, dataset_type):
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    metrics = {
        'Dataset': dataset_type,
        'MSE': mse,
        'MAE': mae,
        'R2 Score': r2
    }
    return metrics

train_metrics = calculate_metrics(actual_train_prices, predicted_train_prices, 'Training')
test_metrics = calculate_metrics(actual_test_prices, predicted_test_prices, 'Testing')

# Combine metrics
metrics_df = pd.DataFrame([train_metrics, test_metrics])

print("Output by Shourya")

# Display metrics
print("\n--- Evaluation Metrics ---")
print(metrics_df)

# Plot predicted vs actual prices for testing data
plt.figure(figsize=(10, 6))
plt.plot(predicted_test_prices, label='Predicted Prices', color='blue')
plt.plot(actual_test_prices, label='Actual Prices', color='orange')
plt.title('Predicted vs Actual Closing Prices (Testing Data)')
plt.xlabel('Test Data Points')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()