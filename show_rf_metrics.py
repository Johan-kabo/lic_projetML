#!/usr/bin/env python3
import pickle
from sklearn.metrics import classification_report, roc_auc_score

# Load pre-trained RF model
with open('models/random_forest_final.pkl', 'rb') as f:
    model = pickle.load(f)

# Load test data
with open('data/prepared_data.pkl', 'rb') as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

print('⚡ RANDOM FOREST')
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]

print(classification_report(y_test, y_pred))
print('ROC-AUC:', roc_auc_score(y_test, y_proba))
