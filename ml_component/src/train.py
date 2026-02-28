"""Train a LogisticRegression pipeline and persist artefacts.

Usage
-----
python -m ml_component.src.train --seed 42 --test-size 0.2 --val-size 0.2 --c 1.0
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_component.src.data_prep import load_and_prepare
from ml_component.src.utils import ensure_dir, save_json


def build_pipeline(C: float = 1.0, seed: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(C=C, max_iter=1000, random_state=seed),
            ),
        ]
    )


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="binary")),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train breast-cancer classifier")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--c", type=float, default=1.0, dest="C")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "outputs"),
    )
    args = parser.parse_args()

    out_dir = os.path.normpath(args.out_dir)
    ensure_dir(out_dir)

    train_df, val_df, _ = load_and_prepare(
        test_size=args.test_size, val_size=args.val_size, seed=args.seed
    )

    feature_cols = [c for c in train_df.columns if c != "target"]
    X_train = train_df[feature_cols].values
    y_train = train_df["target"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["target"].values

    pipeline = build_pipeline(C=args.C, seed=args.seed)
    pipeline.fit(X_train, y_train)

    val_pred = pipeline.predict(X_val)
    val_prob = pipeline.predict_proba(X_val)[:, 1]
    metrics = compute_metrics(y_val, val_pred, val_prob)

    # Save model
    model_path = os.path.join(out_dir, "model.joblib")
    joblib.dump(pipeline, model_path)

    # Save metrics
    save_json(metrics, os.path.join(out_dir, "metrics.json"))

    # Save validation errors
    val_errors = val_df.copy()
    val_errors["predicted"] = val_pred
    val_errors["error"] = (val_errors["predicted"] != val_errors["target"]).astype(int)
    val_errors = val_errors[val_errors["error"] == 1]
    val_errors.to_csv(os.path.join(out_dir, "val_errors.csv"), index=False)

    print(
        f"[train] val accuracy={metrics['accuracy']:.4f} "
        f"f1={metrics['f1']:.4f} "
        f"roc_auc={metrics['roc_auc']:.4f} | "
        f"model saved → {model_path}"
    )


if __name__ == "__main__":
    main()
