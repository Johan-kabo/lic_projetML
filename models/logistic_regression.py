# logistic_regression.py
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Charger les données préparées ---
with open("data/prepared_data.pkl", "rb") as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

# --- 2. Entraîner le modèle ---
log_model = LogisticRegression(max_iter=2000)
log_model.fit(X_train, y_train)

# --- 3. Prédictions et métriques ---
y_pred = log_model.predict(X_test)
y_proba = log_model.predict_proba(X_test)[:,1]

print("⚡ LOGISTIC REGRESSION")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

# --- 4. Matrice de confusion ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Logistic Regression - Matrice de confusion")
plt.xlabel("Prédit")
plt.ylabel("Vrai")
plt.show()
