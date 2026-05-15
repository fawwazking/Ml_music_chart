# clustering.py

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'data' / 'spotify_final_cleaned.csv'
OUTPUT_DIR = BASE_DIR / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Feature engineering
cluster_data = df.groupby('Genre').agg({
    'Streams': 'mean',
    'Position': 'mean',
    'Track Name': 'count',
    'Weeks on Chart': 'mean'
}).reset_index()

cluster_data.columns = [
    'Genre',
    'Avg_Streams',
    'Avg_Position',
    'Song_Count',
    'Avg_Weeks_On_Chart'
]

print(cluster_data)

# Scaling
features = cluster_data[[
    'Avg_Streams',
    'Avg_Position',
    'Song_Count',
    'Avg_Weeks_On_Chart'
]]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Elbow method
inertia = []

K = range(1, 10)

for k in K:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(scaled_features)

    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8,5))

plt.plot(K, inertia, marker='o')

plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'elbow_method.png')
plt.show()

# K-Means clustering
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

cluster_data['Cluster'] = kmeans.fit_predict(
    scaled_features
)

# Silhouette score
score = silhouette_score(
    scaled_features,
    cluster_data['Cluster']
)

print('Silhouette Score:', score)

print(cluster_data)

# Visualization
plt.figure(figsize=(12,6))

sns.scatterplot(
    data=cluster_data,
    x='Avg_Streams',
    y='Song_Count',
    hue='Cluster',
    s=200
)

for i in range(len(cluster_data)):
    plt.text(
        cluster_data['Avg_Streams'][i],
        cluster_data['Song_Count'][i],
        cluster_data['Genre'][i]
    )

plt.title('K-Means Clustering Result')
plt.xlabel('Average Streams')
plt.ylabel('Song Count')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'clustering_result.png')
plt.show()
