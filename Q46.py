import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

data = {
    'CustomerID': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    'Annual_Income': [30000, 45000, 50000, 70000, 35000, 85000, 60000, 40000, 75000, 30000],
    'Spending_Score': [40, 65, 20, 90, 50, 70, 80, 55, 95, 30],
    'Age': [25, 30, 22, 40, 28, 45, 35, 33, 50, 20],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Miami', 
             'Dallas', 'Seattle', 'Denver', 'San Diego', 'Boston']
}

df = pd.DataFrame(data)

X = df[['Annual_Income', 'Spending_Score']]

inertia = []
K = range(1, 10)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of Clusters K')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Visualization
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Annual_Income', y='Spending_Score', hue='Cluster', data=df, palette='Set2', s=100)
plt.title('Customer Segments')
plt.xlabel('Annual Income (USD)')
plt.ylabel('Spending Score')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

score = silhouette_score(X, df['Cluster'])
print(f"Silhouette Score: {score:.2f}")
