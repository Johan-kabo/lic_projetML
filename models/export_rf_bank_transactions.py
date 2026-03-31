#!/usr/bin/env python3
"""
Export RandomForest model trained on bank_transactions_data_2.csv to ONNX.

Loads bank_transactions_data_2.csv, prepares data, trains model, exports to ONNX.
"""
import os
import pickle
import sys
import traceback

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
except ImportError:
    print("Missing dependency: skl2onnx. Install with: pip install skl2onnx")
    sys.exit(1)


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_path = os.path.join(root, "data", "bank_transactions_data_2.csv")
    out_dir = os.path.join(root, "models")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"bank_transactions_data_2.csv not found at: {csv_path}")
        sys.exit(1)

    try:
        print(f"Loading data from: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

        # Option 1: If there's a fraud column, use it; otherwise create synthetic target
        if "Class" in df.columns:
            target_col = "Class"
        elif "IsFraud" in df.columns:
            target_col = "IsFraud"
        elif "Fraud" in df.columns:
            target_col = "Fraud"
        else:
            # Create a synthetic fraud label (randomly for demo)
            print("No fraud column found. Creating synthetic target for demo...")
            df["SyntheticFraud"] = np.random.randint(0, 2, size=len(df))
            target_col = "SyntheticFraud"

        # --- 2. Prepare data ---
        X = df.drop(target_col, axis=1)
        y = df[target_col]

        # Drop non-numeric/non-categorical columns (IDs, text)
        drop_cols = []
        for col in X.columns:
            if X[col].dtype == "object" and col not in ["TransactionType", "Location", "Channel", "CustomerOccupation"]:
                drop_cols.append(col)
        
        if drop_cols:
            print(f"Dropping non-processable columns: {drop_cols}")
            X = X.drop(drop_cols, axis=1)

        # Encode categorical columns
        categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
        label_encoders = {}

        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
            print(f"Encoded: {col}")

        # Normalize numeric columns
        scaler = StandardScaler()
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"\nTraining data shape: {X_train.shape}")
        print(f"Target class distribution: {y_train.value_counts().to_dict()}")

        # --- 3. Train model ---
        print("\nTraining RandomForestClassifier (n_estimators=100)...")
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        # Evaluate
        from sklearn.metrics import classification_report, roc_auc_score
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        try:
            print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
        except Exception as e:
            print(f"Could not compute ROC-AUC: {e}")

        # --- 4. Convert to ONNX ---
        n_features = X_train.shape[1]
        initial_type = [("input", FloatTensorType([None, n_features]))]
        print("\nConverting to ONNX...")
        onnx_model = convert_sklearn(model, initial_types=initial_type)

        out_path = os.path.join(out_dir, "random_forest_bank_transactions.onnx")
        with open(out_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        print(f"\n✅ ONNX model saved to: {out_path}")

    except Exception:
        print("\nAn error occurred:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
