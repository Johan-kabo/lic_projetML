import os
import pickle
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Moteur de Décision Fraude v1.0")

# CORS pour développement : autorise les requêtes preflight (OPTIONS)
# CORS restreint aux UIs locales (dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:8082",
        "http://127.0.0.1:8082",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serveur de Mémoire : Journal de bord des prédictions
db_logs = []

# --- Chargement robuste du modèle ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest_final.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# --- Schéma de données strict (Pydantic) ---
class Transaction(BaseModel):
    Time: float
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float
    
@app.post("/predict-fraud")
def predict(transaction: Transaction):
    try:
        data = pd.DataFrame([transaction.dict()])
        proba = model.predict_proba(data)[0][1]
        
        # --- Logique de Décision Bancaire ---
        # On définit des seuils (Thresholds)
        if proba >= 0.8000:
            decision = "BLOCK"
            action = "Refus automatique et alerte SMS"
        elif proba >= 0.5000:
            decision = "REVIEW"
            action = "Mise en attente et vérification humaine"
        else:
            decision = "APPROVE"
            action = "Transaction autorisée"

        result = {
            "timestamp": datetime.now().isoformat(),
            "model_version": "RF-2026-V1",
            "fraud_probability": round(float(proba), 4),
            "decision": decision,
            "action_required": action,
            "input": transaction.dict()
        }

        # Enregistrer dans le journal de bord (db_logs)
        db_logs.append(result)

        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/get-history")
def get_history(limit: int = 100):
    """Retourne l'historique des prédictions (les plus récentes en premier)."""
    try:
        # renvoyer une copie pour éviter modifications externes
        items = list(db_logs[-limit:])[::-1]
        return {"count": len(db_logs), "history": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get-monitoring")
def get_monitoring():
    """Guichet de Monitoring : Retourne l'état complet du serveur de mémoire.
    
    C'est le point d'accès unique pour le Dashboard:
    - Liste complète des logs (db_logs)
    - Statistiques globales (total, dernière prédiction, etc.)
    - État du système
    """
    try:
        if not db_logs:
            return {
                "status": "OK",
                "total_logs": 0,
                "logs": [],
                "last_prediction": None
            }
        
        stats = {
            "total_approved": sum(1 for log in db_logs if log["decision"] == "APPROVE"),
            "total_review": sum(1 for log in db_logs if log["decision"] == "REVIEW"),
            "total_blocked": sum(1 for log in db_logs if log["decision"] == "BLOCK"),
            "avg_fraud_probability": round(
                sum(log["fraud_probability"] for log in db_logs) / len(db_logs), 4
            )
        }
        
        return {
            "status": "OK",
            "total_logs": len(db_logs),
            "statistics": stats,
            "last_prediction": db_logs[-1] if db_logs else None,
            "logs": list(db_logs)  # Toute la mémoire pour le Dashboard
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/clear")
def clear_logs():
    """Vide le serveur de mémoire `db_logs`.

    Usage rapide: ouvrir `http://127.0.0.1:8000/clear` dans un navigateur.
    """
    try:
        db_logs.clear()
        return {"status": "cleared", "total_logs": 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))