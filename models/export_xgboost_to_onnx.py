#!/usr/bin/env python3
"""Export XGBoost model to ONNX.

Loads prepared data, trains XGBoost, and saves to ONNX format.
"""
import os
import pickle
import sys
import traceback

from xgboost import XGBClassifier

try:
    import onnxmltools
    from onnxmltools.convert.common.data_types import FloatTensorType
except ImportError:
    print("Missing dependency: onnxmltools. Install with: pip install onnxmltools")
    sys.exit(1)


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(root, "data", "prepared_data.pkl")
    out_dir = os.path.join(root, "models")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(data_path):
        print(f"prepared_data.pkl not found at: {data_path}")
        sys.exit(1)

    try:
        print(f"Loading prepared data from: {data_path}")
        with open(data_path, "rb") as f:
            X_train, X_test, y_train, y_test = pickle.load(f)

        # Rename columns to f0, f1, f2... for XGBoost ONNX conversion compatibility
        n_features = X_train.shape[1]
        new_cols = [f"f{i}" for i in range(n_features)]
        X_train.columns = new_cols
        X_test.columns = new_cols

        print("Training XGBClassifier (this may take a moment)...")
        model = XGBClassifier(scale_pos_weight=100, eval_metric="logloss", random_state=42, verbosity=0)
        model.fit(X_train, y_train)

        # Evaluate
        from sklearn.metrics import classification_report, roc_auc_score
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

        # --- Convert to ONNX ---
        n_features = X_train.shape[1]
        initial_type = [("float_input", FloatTensorType([None, n_features]))]
        print("\nConverting to ONNX...")
        onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type)

        out_path = os.path.join(out_dir, "xgboost.onnx")
        onnxmltools.utils.save_model(onnx_model, out_path)

        print(f"\n✅ ONNX model saved to: {out_path}")

    except Exception:
        print("\nAn error occurred:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
