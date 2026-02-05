# detection_fraude.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# --- 1. Charger les données ---
df = pd.read_csv("data/Creditcard.csv")

# --- 2. Séparer features et target ---
X = df.drop("Class", axis=1)
y = df["Class"]

# --- 3. Normalisation (important pour certains modèles) ---
scaler = StandardScaler()
X["Amount"] = scaler.fit_transform(X[["Amount"]])

# --- 4. Split train/test ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Données prêtes ✅")

# --- 5. Sauvegarder les datasets pour les modèles ---
with open("data/prepared_data.pkl", "wb") as f:
    pickle.dump((X_train, X_test, y_train, y_test), f)
