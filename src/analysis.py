import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

######################################
#       Régression linéaire          #      
######################################

# Chargement des données prétraitées
data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
data = pd.read_csv(os.path.join(data_folder, "computer_hardware_final.csv"))

# Séparation variables explicatives (X) et variable cible (y)
X = data.drop(columns=["PRP"])
y = data["PRP"]

# Séparer données en jeu d'entraînement (70%) et test (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entraînement de la régression linéaire
model = LinearRegression()
model.fit(X_train, y_train)

# Prédiction sur l'ensemble test
y_pred = model.predict(X_test)

# Évaluation des résultats
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE (Erreur quadratique moyenne) :", rmse)
print("R² (Coefficient de détermination) :", r2_score(y_test, y_pred))

# Visualisation des résultats
plt.figure(figsize=(10,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Valeurs réelles (PRP)")
plt.ylabel("Valeurs prédites (PRP)")
plt.title("Performance réelle vs prédite (Régression linéaire)", fontsize=14, fontweight='bold', color='green')
plt.grid(True)
plt.show()

# Importance des variables (coefficients)
importance = pd.Series(model.coef_, index=X.columns).sort_values()
importance.plot(kind='barh', title='Importance des variables (Coefficients)', fontsize=12, color='lightblue')
plt.title('Importance des variables (Coefficients)', fontsize=14, fontweight='bold', color='green')
plt.xlabel('Coefficient')
plt.grid(True)
plt.show()
