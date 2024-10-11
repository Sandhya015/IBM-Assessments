import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt

# Step 1: Load the dataset
# Specify the correct delimiter, assuming it's a semicolon in this case
data = pd.read_csv('winequality-red.csv', delimiter=';')  # Update this path as necessary

# Step 2: Clean the column names
data.columns = data.columns.str.replace('"', '').str.strip()  # Remove quotes and extra spaces
print("\nCleaned column names:")
print(data.columns)

# Step 3: Data Cleaning
# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Fill missing values with the mean of numeric columns (if needed)
numeric_data = data.select_dtypes(include='number')
data[numeric_data.columns] = numeric_data.fillna(numeric_data.mean())

# Verify there are no missing values
print("\nMissing values after filling:")
print(data.isnull().sum())

# Step 4: Use OrdinalEncoder for categorical variables
# Check if 'quality' column exists in the DataFrame
if 'quality' in data.columns:
    encoder = OrdinalEncoder(categories=[sorted(data['quality'].unique())])
    data['quality'] = encoder.fit_transform(data[['quality']])
    print("\nEncoded 'quality' values:")
    print(data['quality'].head())
else:
    print("\n'quality' column not found in the DataFrame.")

# Step 5: Convert to NumPy arrays
X = data.drop('quality', axis=1).to_numpy()  # Features
y = data['quality'].to_numpy()  # Target variable

# Step 6: Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Build and Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 8: Evaluate the Model
y_pred = model.predict(X_test)

# Print evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Calculate accuracy based on a threshold
threshold = 0.5  # Define a threshold for accuracy
accuracy = np.mean(np.abs(y_pred - y_test) <= threshold) * 100  # Calculate accuracy
print(f"\nAccuracy (within threshold of {threshold}): {accuracy:.2f}%")

# Optionally, you can print predictions
print("\nPredictions on test set:")
print(y_pred[:10])  # Print first 10 predictions

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)  # 45-degree line
plt.xlabel('Actual Quality')
plt.ylabel('Predicted Quality')
plt.title('Actual vs Predicted Wine Quality')
plt.grid()
plt.show()
