import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Étape 1 : Charger les données originales
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/cpu-performance/machine.data"
data = pd.read_csv(url, header=None)

# Renommer clairement les colonnes
data.columns = [
    "vendor_name", "model_name", "MYCT", "MMIN", "MMAX",
    "CACH", "CHMIN", "CHMAX", "PRP", "ERP"
]

# Affichage rapide
print("Données originales :")
print(data.head())

# Étape 2 : Suppression des variables catégorielles inutiles
data_clean = data.drop(columns=["vendor_name", "model_name"])
print("\nAprès suppression des variables catégorielles (vendor_name, model_name) :")
print(data_clean.head())

# Étape 3 : Suppression de la variable très corrélée ERP (éviter data leakage)
data_clean = data_clean.drop(columns=["ERP"])
print("\nAprès suppression de la variable ERP (corrélée à PRP) :")
print(data_clean.head())

# Étape 4 : Normalisation (standardisation) des variables numériques restantes
# Séparer variables explicatives (X) et cible (y)
X = data_clean.drop(columns=["PRP"])
y = data_clean["PRP"]

# Appliquer la standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Remettre dans un DataFrame pour une meilleure visibilité
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Ajouter la variable cible PRP à la fin
final_data = pd.concat([X_scaled_df, y.reset_index(drop=True)], axis=1)

print("\nAprès normalisation des variables numériques :")
print(final_data.head())

# Sauvegarder le résultat final dans un nouveau fichier CSV propre
#final_data.to_csv("../data/computer_hardware_final.csv", index=False)

# Définir le chemin absolu vers le dossier data à partir du script actuel
data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Vérifier que le dossier data existe, sinon le créer automatiquement
os.makedirs(data_folder, exist_ok=True)

# Maintenant enregistrer correctement tes fichiers dans data
data.to_csv(os.path.join(data_folder, "computer_hardware.csv"), index=False)
data_clean.to_csv(os.path.join(data_folder, "computer_hardware_clean.csv"), index=False)
final_data.to_csv(os.path.join(data_folder, "computer_hardware_final.csv"), index=False)

