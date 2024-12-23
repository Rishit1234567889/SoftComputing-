# Program 7 : Implement linear regression and multi regression for set of data points.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Step 1: Create a dataset
np.random.seed(42)
# Generate data for linear regression
X_single = np.random.rand(100, 1) * 10  # Single feature
y_single = 3.5 * X_single + np.random.randn(100, 1) * 5  # Linear relationship with noise

# Generate data for multiple regression
X_multi = np.random.rand(100, 3) * 10  # Three features
coeffs = [2, -1.5, 3]  # Coefficients for the features
y_multi = np.dot(X_multi, coeffs) + np.random.randn(100) * 5  # Linear relationship with noise

# Step 2: Split the data
X_single_train, X_single_test, y_single_train, y_single_test = train_test_split(
    X_single, y_single, test_size=0.2, random_state=42
)
X_multi_train, X_multi_test, y_multi_train, y_multi_test = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42
)

# Step 3: Train models
lin_reg_single = LinearRegression()
lin_reg_single.fit(X_single_train, y_single_train)

lin_reg_multi = LinearRegression()
lin_reg_multi.fit(X_multi_train, y_multi_train)

# Step 4: Predictions and evaluation
y_single_train_pred = lin_reg_single.predict(X_single_train)
y_single_test_pred = lin_reg_single.predict(X_single_test)

y_multi_train_pred = lin_reg_multi.predict(X_multi_train)
y_multi_test_pred = lin_reg_multi.predict(X_multi_test)

# Step 5: Convert regression outputs to binary classification
# Using the median of the target variable as the threshold for classification
y_single_median = np.median(y_single_train)
y_multi_median = np.median(y_multi_train)

y_single_train_class = (y_single_train > y_single_median).astype(int)
y_single_test_class = (y_single_test > y_single_median).astype(int)

y_multi_train_class = (y_multi_train > y_multi_median).astype(int)
y_multi_test_class = (y_multi_test > y_multi_median).astype(int)

y_single_train_pred_class = (y_single_train_pred > y_single_median).astype(int)
y_single_test_pred_class = (y_single_test_pred > y_single_median).astype(int)

y_multi_train_pred_class = (y_multi_train_pred > y_multi_median).astype(int)
y_multi_test_pred_class = (y_multi_test_pred > y_multi_median).astype(int)

# Step 6: Evaluation metrics for regression (using MSE and R^2)
metrics_single = {
    "Training MSE": mean_squared_error(y_single_train, y_single_train_pred),
    "Testing MSE": mean_squared_error(y_single_test, y_single_test_pred),
    "Training R^2": r2_score(y_single_train, y_single_train_pred),
    "Testing R^2": r2_score(y_single_test, y_single_test_pred),
}

metrics_multi = {
    "Training MSE": mean_squared_error(y_multi_train, y_multi_train_pred),
    "Testing MSE": mean_squared_error(y_multi_test, y_multi_test_pred),
    "Training R^2": r2_score(y_multi_train, y_multi_train_pred),
    "Testing R^2": r2_score(y_multi_test, y_multi_test_pred),
}

# Step 7: Classification metrics (Accuracy, Precision, Recall, F1 score)
metrics_single_classification = {
    "Training Accuracy": accuracy_score(y_single_train_class, y_single_train_pred_class),
    "Testing Accuracy": accuracy_score(y_single_test_class, y_single_test_pred_class),
    "Training Precision": precision_score(y_single_train_class, y_single_train_pred_class),
    "Testing Precision": precision_score(y_single_test_class, y_single_test_pred_class),
    "Training Recall": recall_score(y_single_train_class, y_single_train_pred_class),
    "Testing Recall": recall_score(y_single_test_class, y_single_test_pred_class),
    "Training F1 Score": f1_score(y_single_train_class, y_single_train_pred_class),
    "Testing F1 Score": f1_score(y_single_test_class, y_single_test_pred_class),
}

metrics_multi_classification = {
    "Training Accuracy": accuracy_score(y_multi_train_class, y_multi_train_pred_class),
    "Testing Accuracy": accuracy_score(y_multi_test_class, y_multi_test_pred_class),
    "Training Precision": precision_score(y_multi_train_class, y_multi_train_pred_class),
    "Testing Precision": precision_score(y_multi_test_class, y_multi_test_pred_class),
    "Training Recall": recall_score(y_multi_train_class, y_multi_train_pred_class),
    "Testing Recall": recall_score(y_multi_test_class, y_multi_test_pred_class),
    "Training F1 Score": f1_score(y_multi_train_class, y_multi_train_pred_class),
    "Testing F1 Score": f1_score(y_multi_test_class, y_multi_test_pred_class),
}

# Step 8: Print the metrics
print("Linear Regression (Single Feature) Metrics:")
for metric, value in metrics_single.items():
    print(f"{metric}: {value:.4f}")

print("\nMultiple Regression Metrics:")
for metric, value in metrics_multi.items():
    print(f"{metric}: {value:.4f}")

print("\nLinear Regression (Single Feature) Classification Metrics:")
for metric, value in metrics_single_classification.items():
    print(f"{metric}: {value:.4f}")

print("\nMultiple Regression Classification Metrics:")
for metric, value in metrics_multi_classification.items():
    print(f"{metric}: {value:.4f}")

# Step 9: Plotting
plt.figure(figsize=(12, 6))

# Plot for linear regression
plt.subplot(1, 2, 1)
plt.scatter(X_single, y_single, color="blue", label="Data points")
plt.plot(X_single, lin_reg_single.predict(X_single), color="red", label="Regression line")
plt.title("Linear Regression (Single Feature)")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.legend()

# Plot residuals for multiple regression
residuals = y_multi_test - y_multi_test_pred
plt.subplot(1, 2, 2)
plt.scatter(range(len(residuals)), residuals, color="green")
plt.axhline(y=0, color="red", linestyle="--", linewidth=1.5)
plt.title("Residuals (Multiple Regression)")
plt.xlabel("Index")
plt.ylabel("Residuals")

plt.tight_layout()
plt.show()
