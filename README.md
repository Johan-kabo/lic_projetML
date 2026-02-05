# Détection de Fraude Bancaire par Machine Learning

## Objectif
Détecter automatiquement les transactions frauduleuses en temps réel à l’aide du Machine Learning.

## Dataset
- Credit Card Dataset (transactions réelles)
- Données fortement déséquilibrées

## Modèles testés
- Logistic Regression
- Random Forest ✅
- XGBoost

## Modèle final
Random Forest a été retenu pour ses performances sur les données déséquilibrées.

## Évaluation
- Classification Report
- ROC-AUC
- Matrice de confusion

## Exécution
```bash
python models/random_forest.py
