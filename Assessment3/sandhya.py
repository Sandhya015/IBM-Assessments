# Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster

# Step 1: Data Aggregation and Preprocessing
# Sample Dataset
data = {
    'customer_id': range(1, 101),
    'age': np.random.randint(18, 70, 100),
    'tenure': np.random.randint(1, 12, 100),
    'monthly_spending': np.random.uniform(100, 1000, 100),
    'num_products': np.random.randint(1, 5, 100)
}
df = pd.DataFrame(data)

# Handling Missing Values (if any)
df.fillna(df.mean(), inplace=True)

# Normalizing Numerical Columns
scaler = StandardScaler()
df[['age', 'tenure', 'monthly_spending', 'num_products']] = scaler.fit_transform(df[['age', 'tenure', 'monthly_spending', 'num_products']])

# Visualizing Feature Distributions
plt.figure(figsize=(12, 8))
for i, column in enumerate(['age', 'tenure', 'monthly_spending', 'num_products']):
    plt.subplot(2, 2, i+1)
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column}')
plt.tight_layout()
plt.show()

# Step 2: Clustering Using Hierarchical Clustering
# Selecting Features for Clustering
X = df[['age', 'tenure', 'monthly_spending']]

# Hierarchical Clustering with AgglomerativeClustering
agg_clustering = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
df['cluster'] = agg_clustering.fit_predict(X)

# Step 3: Cluster Evaluation with Dendrogram
# Generating the linkage matrix for the dendrogram
Z = linkage(X, method='ward')
plt.figure(figsize=(10, 6))
dendrogram(Z, truncate_mode='level', p=3)
plt.title('Dendrogram')
plt.xlabel('Customer Index')
plt.ylabel('Distance')
plt.show()

# Step 4: Cluster Profiling
# Summary statistics of clusters
cluster_summary = df.groupby('cluster').agg({
    'age': ['mean', 'median', 'std'],
    'tenure': ['mean', 'median', 'std'],
    'monthly_spending': ['mean', 'median', 'std']
})
print("Cluster Summary Statistics:")
print(cluster_summary)

# Visualizing Clusters using Scatter Plot
sns.pairplot(df, vars=['age', 'tenure', 'monthly_spending'], hue='cluster', palette='viridis')
plt.suptitle('Customer Segments', y=1.02)
plt.show()

# Step 5: Submission
# Save the final dataframe with clusters
df.to_csv('customer_segmentation_with_clusters.csv', index=False)

