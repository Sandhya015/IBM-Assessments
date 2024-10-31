import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Generate some sample data
np.random.seed(10)
x = np.linspace(0, 10, 100)
y = 0.5 * x**2 - x + 3 + np.random.normal(0, 5, size=100)

# Create a DataFrame
data = pd.DataFrame({'x': x, 'y': y})

# Plot using Seaborn with polynomial regression
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.regplot(x='x', y='y', data=data, order=2, scatter_kws={"color": "blue"}, line_kws={"color": "red"})

# Add labels and show the plot
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Polynomial Regression with Seaborn")
plt.show()
