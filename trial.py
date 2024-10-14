# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification

# Step 1: Generate a synthetic dataset with make_classification
X, y = make_classification(n_samples=1000, n_features=2, n_classes=2, 
                           n_informative=2, n_redundant=0, n_clusters_per_class=2, 
                           weights=[0.7, 0.3], random_state=42)

# Step 2: Load the dataset into a Pandas DataFrame for easier handling
df = pd.DataFrame(X, columns=['Age', 'Tenure'])
df['Churn'] = y

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

# Step 8: Use RandomForest Classifier
clf = RandomForestClassifier(random_state=42)

# Step 9a: Option 1 - Use GridSearchCV for hyperparameter tuning (Reduced grid size for faster execution)
param_grid = {
    'n_estimators': [100, 200],         # Reduced the number of estimators
    'max_depth': [10, 20],              # Only two depth levels
    'min_samples_split': [2, 5],        # Two values for min_samples_split
    'min_samples_leaf': [1, 2],         # Reduced number of min_samples_leaf
    'bootstrap': [True]                 # Only True to reduce combinations
}

# GridSearchCV with verbose to monitor progress
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=3)
grid_search.fit(X_train_scaled, y_train_resampled)

# Best classifier from grid search
best_clf = grid_search.best_estimator_

# Step 10: Evaluate the model on the test set
y_pred = best_clf.predict(X_test_scaled)

# Calculate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nGridSearchCV - Model Accuracy: {accuracy * 100:.2f}%")

# Detailed classification report
print("\nClassification Report (GridSearchCV):")
print(classification_report(y_test, y_pred, zero_division=1))

# Step 9b: Option 2 - Use RandomizedSearchCV for faster hyperparameter search
random_search = RandomizedSearchCV(clf, param_grid, cv=5, n_iter=10, scoring='accuracy', random_state=42, n_jobs=-1)
random_search.fit(X_train_scaled, y_train_resampled)

# Best classifier from random search
best_clf_random = random_search.best_estimator_

# Step 10: Evaluate the model on the test set
y_pred_random = best_clf_random.predict(X_test_scaled)

# Calculate model accuracy
accuracy_random = accuracy_score(y_test, y_pred_random)
print(f"\nRandomizedSearchCV - Model Accuracy: {accuracy_random * 100:.2f}%")

# Detailed classification report
print("\nClassification Report (RandomizedSearchCV):")
print(classification_report(y_test, y_pred_random, zero_division=1))
