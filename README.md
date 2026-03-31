# Détection de Fraude Bancaire par Machine Learning

## Objectif
Détecter automatiquement les transactions frauduleuses en temps réel à l'aide du Machine Learning.

## Dataset
- Credit Card Dataset (transactions réelles)
- Données fortement déséquilibrées

## Modèles testés
- Logistic Regression
- Random Forest 
- XGBoost

## Modèle final
Random Forest a été retenu pour ses performances sur les données déséquilibrées.

## Évaluation
- Classification Report
- ROC-AUC
- Matrice de confusion

## Architecture
- **API FastAPI** : Serveur REST pour prédictions temps réel
- **Modèles ONNX** : Export optimisé pour déploiement
- **Monitoring** : Dashboard en temps réel

## Installation
```bash
pip install -r requirements.txt
```

## Exécution
```bash
# Préparation données
python detection_fraude.py

# Entraînement modèle
python models/random_forest.py

# Lancement API
python -m uvicorn API.api:app --reload --host 0.0.0.0 --port 8000
```

## Endpoints API
- `POST /predict-fraud` : Prédiction fraude
- `GET /get-monitoring` : Dashboard monitoring
- `GET /docs` : Documentation Swagger
