#!/usr/bin/env python3
"""Export RandomForest model to ONNX.

Loads prepared data from `data/prepared_data.pkl`, trains a RandomForest
classifier (same params as the training script) and saves an ONNX file to
`models/random_forest.onnx`.
"""
import os
import pickle
import sys
import traceback

from sklearn.ensemble import RandomForestClassifier


def main():
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except Exception:
        print("Missing dependency: skl2onnx. Install with: pip install skl2onnx")
        sys.exit(1)

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

        print("Training RandomForestClassifier (n_estimators=100)...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        n_features = X_train.shape[1]
        initial_type = [("input", FloatTensorType([None, n_features]))]
        print("Converting to ONNX (this may take a moment)...")
        onnx_model = convert_sklearn(model, initial_types=initial_type)

        out_path = os.path.join(out_dir, "random_forest.onnx")
        with open(out_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        print(f"✅ ONNX model saved to: {out_path}")
    except Exception:
        print("An error occurred during export:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
