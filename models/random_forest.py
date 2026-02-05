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
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# --- Prédictions et métriques ---
y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)[:,1]

print("⚡ RANDOM FOREST")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))



# --- Sauvegarde du modèle ---
with open("../random_forest_final.pkl", "wb") as f:
    pickle.dump(rf_model, f)

print("✅ Modèle sauvegardé avec succès")

# --- Matrice de confusion ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Random Forest - Matrice de confusion")
plt.xlabel("Prédit")
plt.ylabel("Vrai")
plt.show()

