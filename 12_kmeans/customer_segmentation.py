import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Load dataset
print("--- Loading Dataset ---")
df = pd.read_csv('Mall_Customers.csv')

# 2. Show dataset info
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nFirst 5 rows:")
print(df.head())

# 3. Drop CustomerID
df_processed = df.drop('CustomerID', axis=1)

# 4. Select features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# 5. Apply StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Run KMeans for K = 1 to 10
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# 7. Plot Elbow Curve
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertia, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.grid(True)
plt.savefig('elbow_curve.png')
print("\nElbow curve saved as 'elbow_curve.png'")

# 8. Print Inertia Table
inertia_table = pd.DataFrame({'K': K_range, 'Inertia': inertia})
print("\nInertia Table:")
print(inertia_table)

# 9. Choose best K (Typically 5 for this dataset)
best_k = 5
print(f"\nChosen K: {best_k}")

# 10. Train final KMeans model
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['Cluster'] = kmeans_final.fit_predict(X_scaled)

# 11. Plot clusters
plt.figure(figsize=(12, 8))
colors = ['red', 'blue', 'green', 'orange', 'purple']
for i in range(best_k):
    cluster_data = df[df['Cluster'] == i]
    plt.scatter(cluster_data['Annual Income (k$)'], 
                cluster_data['Spending Score (1-100)'], 
                s=100, label=f'Cluster {i}')

# Plot centers
centers_scaled = kmeans_final.cluster_centers_
centers = scaler.inverse_transform(centers_scaled)
plt.scatter(centers[:, 0], centers[:, 1], s=300, c='black', marker='X', label='Centroids')

plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.savefig('customer_clusters.png')
print("Cluster visualization saved as 'customer_clusters.png'")

# 12. Interpret each cluster
def map_label(cluster_id, df_subset):
    avg_income = df_subset['Annual Income (k$)'].mean()
    avg_spending = df_subset['Spending Score (1-100)'].mean()
    
    # Heuristic mapping based on centers
    if avg_income > 70 and avg_spending > 60:
        return 'Premium Customers'
    elif avg_income > 70 and avg_spending < 40:
        return 'Careful Customers'
    elif avg_income < 40 and avg_spending > 60:
        return 'Impulsive Customers'
    elif avg_income < 40 and avg_spending < 40:
        return 'Budget Customers'
    else:
        return 'Standard Customers'

cluster_labels = {}
for i in range(best_k):
    cluster_labels[i] = map_label(i, df[df['Cluster'] == i])

df['Customer_Type'] = df['Cluster'].map(cluster_labels)

# 13. Save final dataset
df.to_csv('segmented_customers.csv', index=False)
print("\nFinal segmented dataset saved as 'segmented_customers.csv'")

# 14. Print final output summary
print("\n--- Final Output Summary ---")
print("\nCluster Counts:")
print(df['Customer_Type'].value_counts())

summary = df.groupby('Customer_Type')[['Annual Income (k$)', 'Spending Score (1-100)']].mean()
print("\nMean Income and Spending Score per Cluster:")
print(summary)
