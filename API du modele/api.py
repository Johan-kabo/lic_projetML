from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI(title="Fraud Detection API")

with open("models/random_forest_final.pkl", "rb") as f:
    model = pickle.load(f)

@app.post("/predict")
def predict(transaction: dict):
    df = pd.DataFrame([transaction])
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]

    return {
        "fraud": bool(pred),
        "fraud_probability": round(proba, 4)
    }
