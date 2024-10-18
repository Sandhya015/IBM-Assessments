import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# House pricing dataset (replace with a larger dataset for higher efficiency)
data = {
    'size': [1500, 2000, 2500, 1800, 2200, 3000, 3500, 4000, 2700, 3200],
    'num_bedrooms': [3, 4, 4, 3, 4, 5, 5, 6, 4, 5],
    'num_bathrooms': [2, 3, 3, 2, 3, 4, 4, 5, 3, 4],
    'age': [10, 15, 8, 12, 5, 2, 1, 3, 9, 6],
    'price': [300000, 450000, 500000, 380000, 450000, 700000, 850000, 950000, 550000, 750000]
}

df = pd.DataFrame(data)

# Split the data into features (X) and target (y)
X = df[['size', 'num_bedrooms', 'num_bathrooms', 'age']]
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Linear Regression
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

# 2. Decision Tree Regression
model_dt = DecisionTreeRegressor(random_state=42)
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_test)

mae_dt = mean_absolute_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))

# 3. Random Forest Regression
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

# Display results
print(f"Linear Regression - MAE: {mae_lr}, RMSE: {rmse_lr}")
print(f"Decision Tree - MAE: {mae_dt}, RMSE: {rmse_dt}")
print(f"Random Forest - MAE: {mae_rf}, RMSE: {rmse_rf}")

# Plot efficiency comparison
models = ['Linear Regression', 'Decision Tree', 'Random Forest']
maes = [mae_lr, mae_dt, mae_rf]
rmses = [rmse_lr, rmse_dt, rmse_rf]

# Plot MAE
plt.bar(models, maes, color=['blue', 'green', 'red'])
plt.xlabel('Models')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('Model Comparison (MAE)')
plt.show()

# Plot RMSE
plt.bar(models, rmses, color=['blue', 'green', 'red'])
plt.xlabel('Models')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.title('Model Comparison (RMSE)')
plt.show()
