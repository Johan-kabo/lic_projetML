import pickle
import pandas as pd

# Charger le modèle
with open("models/random_forest_final.pkl", "rb") as f:
    model = pickle.load(f)

# Exemple de transaction
transaction = pd.DataFrame([{
    "Time": 100000,
    "V1": -1.359,
    "V2": -0.072,
    "V3": 2.536,
    "V4": 1.378,
    "V5": -0.338,
    "V6": 0.462,
    "V7": 0.239,
    "V8": 0.098,
    "V9": 0.364,
    "V10": 0.090,
    "V11": -0.551,
    "V12": -0.617,
    "V13": -0.991,
    "V14": -0.311,
    "V15": 1.468,
    "V16": -0.470,
    "V17": 0.208,
    "V18": 0.025,
    "V19": 0.404,
    "V20": 0.251,
    "V21": -0.018,
    "V22": 0.277,
    "V23": -0.110,
    "V24": 0.066,
    "V25": 0.128,
    "V26": -0.189,
    "V27": 0.133,
    "V28": -0.021,
    "Amount": 149.62
}])

# Prédiction
prediction = model.predict(transaction)[0]
probability = model.predict_proba(transaction)[0][1]

print("🚨 FRAUDE" if prediction == 1 else "✅ TRANSACTION SAINE")
print("Probabilité de fraude :", round(probability, 3))
