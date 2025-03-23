import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os

######################################
# Analyse en Composantes Principales #      
######################################

# Charger les données normalisées finales
data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
data = pd.read_csv(os.path.join(data_folder, "computer_hardware_final.csv"))

# Variables explicatives uniquement (sans la cible PRP)
X = data.drop(columns=["PRP"])

# Réaliser l'ACP en choisissant 2 composantes principales
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Créer un DataFrame pour visualiser clairement les résultats
pca_df = pd.DataFrame(data=X_pca, columns=["Composante Principale 1", "Composante Principale 2"])

# Visualiser les résultats de l'ACP
plt.figure(figsize=(10, 7))
sns.scatterplot(x="Composante Principale 1", y="Composante Principale 2", data=pca_df)
plt.title("Visualisation des données après ACP", fontsize=14, fontweight='bold', color='purple')
plt.xlabel("Composante Principale 1")
plt.ylabel("Composante Principale 2")
plt.grid(True)
plt.show()

# Afficher la variance expliquée par chaque composante
explained_variance = pca.explained_variance_ratio_
print(f"Variance expliquée par la Composante Principale 1 : {explained_variance[0]*100:.2f}%")
print(f"Variance expliquée par la Composante Principale 2 : {explained_variance[1]*100:.2f}%")

# Visualiser clairement la variance expliquée
plt.figure(figsize=(8, 5))
sns.barplot(x=["PC1", "PC2"], y=explained_variance, palette='mako')
plt.title("Variance expliquée par chaque composante principale", fontsize=14, fontweight='bold', color='purple')
plt.ylabel("Proportion de variance expliquée")
plt.xlabel("Composantes principales")
plt.grid(True)
plt.show()


# Afficher les coefficients des composantes principales
loadings = pd.DataFrame(pca.components_.T, columns=['CP1', 'CP2'], index=X.columns)
print("Coefficients des composantes principales :")
print(loadings)
