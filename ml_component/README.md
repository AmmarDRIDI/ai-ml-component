# ml_component

A reproducible machine-learning component (M1 / CPU-friendly) that trains and evaluates a binary classifier on the built-in `breast_cancer` dataset from scikit-learn.

---

## Directory layout

```
ml_component/
├── README.md
├── requirements.txt
├── .gitignore
├── outputs/          # created at runtime (git-ignored)
├── src/
│   ├── __init__.py
│   ├── data_prep.py  # load, clean, split
│   ├── train.py      # train & save artefacts
│   ├── evaluate.py   # evaluate on test split
│   └── utils.py      # shared helpers
└── tests/
    ├── __init__.py
    └── test_data_prep.py
```

---

## Setup

```bash
# From the repository root
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r ml_component/requirements.txt
```

---

## Train

Run from the **repository root**:

```bash
python -m ml_component.src.train --seed 42 --test-size 0.2 --val-size 0.2 --c 1.0
```

Artefacts written to `ml_component/outputs/`:

| File | Description |
|------|-------------|
| `model.joblib` | Serialised `Pipeline(StandardScaler, LogisticRegression)` |
| `metrics.json` | Validation accuracy / F1 / ROC-AUC |
| `val_errors.csv` | Mis-classified validation rows |

---

## Evaluate

```bash
python -m ml_component.src.evaluate \
    --seed 42 \
    --model-path ml_component/outputs/model.joblib \
    --out-dir ml_component/outputs
```

Artefacts written to `ml_component/outputs/`:

| File | Description |
|------|-------------|
| `test_metrics.json` | Test accuracy / F1 / ROC-AUC |
| `confusion_matrix.png` | Confusion-matrix plot |

---

## Test

Run from the **repository root**:

```bash
pytest -q
```

Or from inside `ml_component/`:

```bash
cd ml_component
pytest -q
```
