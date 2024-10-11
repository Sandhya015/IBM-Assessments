# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Step 1: Create a sample customer dataset
data = {
    'CustomerID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Age': [23, 45, 35, 50, 40, 60, 22, 32, 43, 55],
    'Tenure': [12, 45, 30, 50, 40, 30, 15, 25, 35, 10],
    'Churn': [1, 0, 0, 1, 0, 1, 0, 1, 0, 0]  # 1 indicates churned, 0 indicates not churned
}

# Step 2: Load the dataset into a Pandas DataFrame
df = pd.DataFrame(data)

# Step 3: Data Aggregation - Calculate average age and tenure for churned and non-churned customers
aggregated_data = df.groupby('Churn').agg({'Age': 'mean', 'Tenure': 'mean'})
print("Aggregated Data (Average Age and Tenure):")
print(aggregated_data)

# Step 4: Split the dataset into Features (X) and Target variable (y)
X = df[['Age', 'Tenure']]  # Features: Age and Tenure
y = df['Churn']  # Target: Churn

# Step 5: Split the data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Step 6: Handle imbalanced data using SMOTE (Reducing n_neighbors to avoid error)
smote = SMOTE(random_state=42, k_neighbors=2)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Step 7: Feature scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Step 8: Use Logistic Regression
clf = LogisticRegression(random_state=42)

# Perform cross-validation for better model evaluation
cv_scores = cross_val_score(clf, X_train_scaled, y_train_resampled, cv=5, scoring='accuracy')
print(f"\nCross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean Cross-Validation Accuracy: {np.mean(cv_scores):.2f}")

# Train the model on the training data
clf.fit(X_train_scaled, y_train_resampled)

# Step 9: Evaluate the model on the test set
y_pred = clf.predict(X_test_scaled)

# Calculate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=1))  # Set zero_division to avoid precision warnings

# Conclusion: Model Performance
print("Model trained and evaluated successfully!")
