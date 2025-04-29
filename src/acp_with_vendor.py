import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

####################################################
# Analyse en Composantes Principales Avec Vendeur  #      
####################################################

# --- Définir les chemins ---
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_path, "data", "computer_hardware.csv")
figures_path = os.path.join(base_path, "figures")
os.makedirs(figures_path, exist_ok=True)

# Chargement du fichier brut avec saut de la première ligne
data = pd.read_csv(data_path, header=None, skiprows=1)

# Nommer correctement les colonnes
data.columns = [
    "vendor_name", "model_name", "MYCT", "MMIN", "MMAX",
    "CACH", "CHMIN", "CHMAX", "PRP", "ERP"
]


# --- Encodage one-hot de 'vendor_name' ---
data_encoded = pd.get_dummies(data, columns=["vendor_name"], drop_first=True)

# --- Suppression des colonnes non pertinentes ---
data_encoded = data_encoded.drop(columns=["model_name", "ERP"])

# --- Séparer les variables explicatives et la cible ---
X = data_encoded.drop(columns=["PRP"])
y = data_encoded["PRP"]

# --- Normalisation des variables ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- ACP avec 2 composantes principales ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# --- Visualisation de l'ACP ---
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1])
plt.title("ACP incluant la variable 'vendor_name' (encodée)", fontsize=14, fontweight='bold', color='green')
plt.xlabel("Composante Principale 1")
plt.ylabel("Composante Principale 2")
plt.grid(True)
plt.tight_layout()

# --- Enregistrement du graphique ---
output_path = os.path.join(figures_path, "acp_with_vendor.png")
plt.savefig(output_path)
plt.show()

# --- Variance expliquée ---
explained_var = pca.explained_variance_ratio_
print(f"Variance expliquée par CP1 : {explained_var[0]*100:.2f}%")
print(f"Variance expliquée par CP2 : {explained_var[1]*100:.2f}%")

print(f"\n Graphique enregistré dans : {output_path}")
