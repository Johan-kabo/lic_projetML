# random_forest.py
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Charger les données préparées ---
with open("data/prepared_data.pkl", "rb") as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

# --- Entraîner le modèle ---
rf_model = RandomForestClassifier(n_estimators=1000, random_state=42)
rf_model.fit(X_train, y_train)

# --- Prédictions et métriques ---
y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)[:,1]

print("⚡ RANDOM FOREST")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

# import os
# print("Le fichier sera ici :", os.path.abspath("random_forest_final.pkl"))

# --- Sauvegarde du modèle dans le bon dossier ---
import os

# On s'assure que le dossier models existe
os.makedirs("../models", exist_ok=True)

# On sauvegarde au bon endroit
save_path = "../models/random_forest_final.pkl"

with open(save_path, "wb") as f:
    pickle.dump(rf_model, f)

print(f"✅ Modèle sauvegardé avec succès dans : {save_path}")
print(f"📦 Taille du fichier : {os.path.getsize(save_path) / 1024**2:.2f} Mo")


# --- Matrice de confusion ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False)
plt.title("Random Forest - Matrice de confusion")
plt.xlabel("Prédit")
plt.ylabel("Vrai")
plt.show()

