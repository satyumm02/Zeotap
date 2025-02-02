import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load datasets
customers = pd.read_csv("Customers.csv")
transactions = pd.read_csv("Transactions.csv")

# Data Preprocessing
# Convert 'SignupDate' to tenure in days
customers['SignupDate'] = pd.to_datetime(customers['SignupDate'])
customers['TenureDays'] = (pd.Timestamp.now() - customers['SignupDate']).dt.days

# Aggregate transaction data by CustomerID
transactions_agg = transactions.groupby('CustomerID').agg({
    'Quantity': 'sum',
    'TotalValue': 'sum',
    'TransactionID': 'count'
}).rename(columns={'TransactionID': 'TransactionCount'}).reset_index()

# Merge customer and transaction data
merged_data = pd.merge(customers, transactions_agg, on='CustomerID', how='inner')

# One-hot encode 'Region'
encoder = OneHotEncoder(sparse_output=False)

region_encoded = encoder.fit_transform(merged_data[['Region']])
region_encoded_df = pd.DataFrame(region_encoded, columns=encoder.get_feature_names_out(['Region']))
merged_data = pd.concat([merged_data, region_encoded_df], axis=1)

# Select features for clustering
features = ['TotalValue', 'Quantity', 'TransactionCount', 'TenureDays'] + list(region_encoded_df.columns)
X = merged_data[features]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means Clustering
best_k = 5  # Based on Davies-Bouldin Index evaluation
kmeans = KMeans(n_clusters=best_k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
merged_data['Cluster'] = clusters

# Evaluate Clustering Metrics
db_index = davies_bouldin_score(X_scaled, clusters)
silhouette_avg = silhouette_score(X_scaled, clusters)

print(f"Davies-Bouldin Index: {db_index}")
print(f"Silhouette Score: {silhouette_avg}")

# Visualization: Pairplot of Clusters
sns.pairplot(merged_data, vars=['TotalValue', 'Quantity', 'TransactionCount', 'TenureDays'], hue='Cluster', palette='Set2')
plt.suptitle("Cluster Visualization", y=1.02)
plt.show()
