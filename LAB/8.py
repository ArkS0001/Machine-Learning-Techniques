import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load data
df = pd.read_csv('data.csv')
X = df.values

# Apply k-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
kmeans_labels = kmeans.predict(X)
kmeans_score = silhouette_score(X, kmeans_labels)
print("k-Means Silhouette Score:", kmeans_score)

# Apply EM algorithm (Gaussian Mixture Model)
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)
gmm_labels = gmm.predict(X)
gmm_score = silhouette_score(X, gmm_labels)
print("EM Algorithm Silhouette Score:", gmm_score)
