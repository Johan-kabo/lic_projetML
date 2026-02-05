# xgboost_model.py
import pickle
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Charger les données préparées ---
with open("data/prepared_data.pkl", "rb") as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

# --- Entraîner le modèle ---
xgb_model = XGBClassifier(scale_pos_weight=100, eval_metric="logloss")
xgb_model.fit(X_train, y_train)

# --- Prédictions et métriques ---
y_pred = xgb_model.predict(X_test)
y_proba = xgb_model.predict_proba(X_test)[:,1]

print("⚡ XGBOOST")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

# --- Matrice de confusion ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("XGBoost - Matrice de confusion")
plt.xlabel("Prédit")
plt.ylabel("Vrai")
plt.show()
