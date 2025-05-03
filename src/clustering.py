import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Définir les chemins
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_path, "data", "computer_hardware_final.csv")
figures_path = os.path.join(base_path, "figures")
os.makedirs(figures_path, exist_ok=True)

# Charger les données
data = pd.read_csv(data_path)

# Séparer les variables explicatives
X = data.drop(columns=["PRP"])

# Appliquer le clustering K-means (ex : 3 clusters)
kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(X)

# Réduction de dimension avec PCA (pour affichage)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualisation des clusters dans le plan PCA
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette="Set2", s=60)
plt.title("Clustering des ordinateurs dans l'espace PCA", fontsize=13, fontweight='bold')
plt.xlabel("Composante Principale 1")
plt.ylabel("Composante Principale 2")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(figures_path, "clustering_pca.png"))
plt.show()
