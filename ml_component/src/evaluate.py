"""Evaluate a saved model on the held-out test split.

Usage
-----
python -m ml_component.src.evaluate \\
    --seed 42 \\
    --model-path ml_component/outputs/model.joblib \\
    --out-dir ml_component/outputs
"""
import argparse
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from ml_component.src.data_prep import load_and_prepare
from ml_component.src.utils import ensure_dir, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate breast-cancer classifier")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "outputs"),
    )
    args = parser.parse_args()

    out_dir = os.path.normpath(args.out_dir)
    ensure_dir(out_dir)

    _, _, test_df = load_and_prepare(
        test_size=args.test_size, val_size=args.val_size, seed=args.seed
    )

    feature_cols = [c for c in test_df.columns if c != "target"]
    X_test = test_df[feature_cols].values
    y_test = test_df["target"].values

    pipeline = joblib.load(args.model_path)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, average="binary")),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }
    save_json(metrics, os.path.join(out_dir, "test_metrics.json"))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    ax.set_title("Confusion Matrix – Test Split")
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    fig.savefig(cm_path, bbox_inches="tight")
    plt.close(fig)

    print(
        f"[evaluate] test accuracy={metrics['accuracy']:.4f} "
        f"f1={metrics['f1']:.4f} "
        f"roc_auc={metrics['roc_auc']:.4f} | "
        f"confusion matrix saved → {cm_path}"
    )


if __name__ == "__main__":
    main()
