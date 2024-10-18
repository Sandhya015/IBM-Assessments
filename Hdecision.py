import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Manually creating a dataset using a dictionary
data = {
    'size': [1500, 2500, 1800, 2200, 3000, 3500, 2800, 2000, 2700, 3200],  # Size in sqft
    'num_bedrooms': [3, 4, 3, 3, 5, 6, 4, 3, 4, 5],  # Number of bedrooms
    'num_bathrooms': [2, 3, 2, 2, 4, 5, 3, 2, 3, 4],  # Number of bathrooms
    'age': [10, 5, 20, 15, 3, 2, 7, 8, 6, 4],  # Age of house in years
    'price': [300000, 500000, 350000, 450000, 600000, 700000, 550000, 400000, 520000, 650000]  # Price in USD
}

# Convert the dictionary into a pandas DataFrame
df = pd.DataFrame(data)

# Splitting features (X) and target (y)
X = df[['size', 'num_bedrooms', 'num_bathrooms', 'age']]
y = df['price']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Regression Model
model_dt = DecisionTreeRegressor(random_state=42)
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_test)

# Evaluate performance
mae_dt = mean_absolute_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))

# Output the performance metrics
print(f"Decision Tree Regression - MAE: {mae_dt}, RMSE: {rmse_dt}")

# Residual Plot
residuals_dt = y_test - y_pred_dt
plt.figure(figsize=(10, 5))
plt.scatter(y_pred_dt, residuals_dt, color='blue')
plt.axhline(0, linestyle='--', color='red')
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.title('Residual Plot: Decision Tree Regression - Predicted Prices vs Residuals')
plt.savefig('residual_plot_decision_tree_regression.png')  # Save the residual plot as a PNG file

# Histogram of Residuals
plt.figure(figsize=(10, 5))
plt.hist(residuals_dt, bins=10, color='blue', edgecolor='black')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals: Decision Tree Regression')
plt.savefig('histogram_residuals_decision_tree_regression.png')  # Save histogram as a PNG file

# Box Plot of Actual vs Predicted Prices
plt.figure(figsize=(10, 5))
plt.boxplot([y_test, y_pred_dt], labels=['Actual Prices', 'Predicted Prices'])
plt.ylabel('Prices')
plt.title('Box Plot: Actual vs Predicted Prices - Decision Tree Regression')
plt.savefig('boxplot_actual_vs_predicted_decision_tree_regression.png')  # Save box plot as a PNG file
