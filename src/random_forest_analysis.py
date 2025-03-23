import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

######################################
#       Random Forest Regressor      #      
######################################

# Chargement des données prétraitées
data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
data = pd.read_csv(os.path.join(data_folder, "computer_hardware_final.csv"))

# Séparation variables explicatives (X) et variable cible (y)
X = data.drop(columns=["PRP"])
y = data["PRP"]

# Séparer les données en ensemble d'entraînement (70%) et test (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entraînement du modèle Random Forest
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Prédiction avec le modèle Random Forest
y_pred_rf = model_rf.predict(X_test)

# Évaluation des résultats
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print("Random Forest Regressor - RMSE :", rmse_rf)
print("Random Forest Regressor - R² :", r2_rf)

# Visualisation des résultats réels vs prédits (Random Forest)
plt.figure(figsize=(10,6))
sns.scatterplot(x=y_test, y=y_pred_rf, color="green")
plt.xlabel("Valeurs réelles (PRP)")
plt.ylabel("Valeurs prédites (PRP)")
plt.title("Performance réelle vs prédite (Random Forest Regressor)", fontsize=14, fontweight='bold', color='purple')
plt.grid(True)
plt.show()

# Importance des variables selon Random Forest
importances_rf = pd.Series(model_rf.feature_importances_, index=X.columns).sort_values()
importances_rf.plot(kind='barh', title='Importance des variables (Random Forest)', fontsize=12, color='lightblue')
plt.title('Importance des variables (Random Forest)', fontsize=14, fontweight='bold', color='purple')
plt.xlabel('Importance')
plt.grid(True)
plt.show()
