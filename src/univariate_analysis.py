import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import os

# Définir le chemin vers les données finales
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_path, "data", "computer_hardware_final.csv")
figures_path = os.path.join(base_path, "figures")
os.makedirs(figures_path, exist_ok=True)

# Charger les données
data = pd.read_csv(data_path)

# Statistiques descriptives
print("📊 Statistiques descriptives :\n")
print(data.describe())

# Sauvegarder les statistiques dans un fichier texte
stats_path = os.path.join(base_path, "data", "statistiques_univariees.txt")
with open(stats_path, "w") as f:
    f.write(data.describe().to_string())

# Liste des variables (sans la cible)
variables = data.drop(columns=["PRP"]).columns

# Boîte à moustaches (boxplot) global
plt.figure(figsize=(12, 6))
sns.boxplot(data=data[variables])
plt.title("Boxplot des variables explicatives (normalisées)", fontsize=14, fontweight='bold', color='green')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(figures_path, "boxplot_variables.png"))
plt.show()

# Histogrammes pour chaque variable
for var in variables:
    plt.figure(figsize=(7, 4))
    sns.histplot(data[var], kde=True, color='skyblue')
    plt.title(f"Distribution de {var}", fontsize=12, fontweight='bold', color='darkblue')
    plt.xlabel(var)
    plt.ylabel("Fréquence")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, f"hist_{var}.png"))
    plt.close()

print("\n Analyse univariée terminée.")
print(f"Résultats enregistrés dans : {figures_path}")
images_paths = [
    "figures/hist_MYCT.png",
    "figures/hist_MMIN.png",
    "figures/hist_MMAX.png",
    "figures/hist_CACH.png",
    "figures/hist_CHMIN.png",
    "figures/hist_CHMAX.png"
]

# Créer une figure avec 2 lignes et 3 colonnes
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# Boucle pour charger et afficher chaque image
for ax, img_path in zip(axs.flatten(), images_paths):
    img = mpimg.imread(img_path)
    ax.imshow(img)
    ax.axis('off')  # Cacher les axes

plt.tight_layout()
plt.savefig("figures/histogrammes_groupes.png")
plt.show()