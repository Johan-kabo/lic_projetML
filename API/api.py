import os
import pickle
import pandas as pd
from fastapi import FastAPI

app = FastAPI(title="Fraud Detection API")

# --- Gestion dynamique du chemin ---
# 1. On récupère le chemin du dossier où se trouve api.py (le dossier API)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. On remonte d'un cran pour arriver à la racine (ProjetML)
root_dir = os.path.dirname(current_dir)
# 3. On pointe vers le fichier dans le dossier models
MODEL_PATH = os.path.join(root_dir, "models", "random_forest_final.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return {"message": "Bienvenue sur l'API de détection de fraude. Allez sur /docs pour tester !"}

from pydantic import BaseModel

class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    # ... liste toutes tes variables ici ...
    V28: float
    Amount: float

@app.post("/predict")
def predict(transaction: dict):
    df = pd.DataFrame([transaction])
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]

    return {
        "fraud": bool(pred),
        "fraud_probability": round(float(proba), 4)
    }