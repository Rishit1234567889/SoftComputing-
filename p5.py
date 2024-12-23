# Practical 6: Plot the correlation plot on dataset and visualize giving an overview of relationships among data on soya bins data. Analysis of covariance: variance (ANOVA), if data have categorial variables on iris data.
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, r2_score

# Updated file path for the Soybean dataset
soybean_file_path = r"/content/soybean-large.data"

# Check if the file exists
if not os.path.exists(soybean_file_path):
    print(f"Error: The file at {soybean_file_path} was not found.")
    exit()

# Load the Soybean dataset
soybean_data = pd.read_csv(soybean_file_path, header=None)
print("Soybean dataset loaded successfully.\n")

# Assigning actual feature names to the dataset
feature_names = [
    'date', 'plant-stand', 'precipitation', 'temperature', 'hail', 'crop-hist', 'area-damaged',
    'severity', 'seed-tmt', 'germination', 'plant-growth', 'leaves', 'leafspots-halo',
    'leafspots-marg', 'leafspot-size', 'leaf-shread', 'leaf-malf', 'leaf-mild', 'stem',
    'lodging', 'stem-cankers', 'canker-lesion', 'fruiting-bodies', 'external decay', 'mycelium',
    'int-discolor', 'sclerotia', 'fruit-pods', 'fruit spots', 'seed', 'mold-growth',
    'seed-discolor', 'seed-size', 'shriveling', 'roots', 'disease-class'
]

# Assign the column names to the dataframe
soybean_data.columns = feature_names

# Convert all categorical columns to numerical using label encoding
label_encoder = LabelEncoder()
soybean_encoded = soybean_data.apply(lambda col: label_encoder.fit_transform(col) if col.dtype == 'object' else col)

# Correlation Heatmap
correlation_matrix = soybean_encoded.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title("Correlation Heatmap of Soybean Dataset")
plt.show()

# Assuming the last column is the target (disease class)
target_column = 'disease-class'
features = soybean_encoded.columns[:-1]

# Split the dataset into training and testing sets
X = soybean_encoded[features]
y = soybean_encoded[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluate the model
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Metrics for Training Data
train_precision = precision_score(y_train, y_train_pred, average='weighted', zero_division=0)
train_recall = recall_score(y_train, y_train_pred, average='weighted')
train_f1 = f1_score(y_train, y_train_pred, average='weighted')
train_r2 = r2_score(y_train, y_train_pred)

print(f"\nTraining Metrics:")
print(f"Accuracy: {train_accuracy * 100:.2f}%")
print(f"Precision: {train_precision:.2f}")
print(f"Recall: {train_recall:.2f}")
print(f"F1-Score: {train_f1:.2f}")
print(f"R²: {train_r2:.2f}")
print("Training Confusion Matrix:")
print(confusion_matrix(y_train, y_train_pred))

# Metrics for Testing Data
test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
test_recall = recall_score(y_test, y_test_pred, average='weighted')
test_f1 = f1_score(y_test, y_test_pred, average='weighted')
test_r2 = r2_score(y_test, y_test_pred)

print(f"\nTesting Metrics:")
print(f"Accuracy: {test_accuracy * 100:.2f}%")
print(f"Precision: {test_precision:.2f}")
print(f"Recall: {test_recall:.2f}")
print(f"F1-Score: {test_f1:.2f}")
print(f"R²: {test_r2:.2f}")
print("Testing Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

# Feature Importance
feature_importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
feature_importances.plot(kind='bar', color='skyblue')
plt.title("Feature Importances")
plt.ylabel("Importance Score")
plt.show()