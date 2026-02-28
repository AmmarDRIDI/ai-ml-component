# ai-ml-component

A repository containing a reproducible machine-learning component for binary classification.

## ML Component

The ML component is located in [`ml_component/`](ml_component/README.md).  
It trains and evaluates a logistic-regression classifier on the scikit-learn `breast_cancer` dataset and is compatible with M1/ARM and standard x86 hardware.

### Quick start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r ml_component/requirements.txt

# 3. Train the model (artefacts written to ml_component/outputs/)
python -m ml_component.src.train --seed 42 --test-size 0.2 --val-size 0.2 --c 1.0

# 4. Evaluate on the held-out test split
python -m ml_component.src.evaluate \
    --seed 42 \
    --model-path ml_component/outputs/model.joblib \
    --out-dir ml_component/outputs

# 5. Run tests
pytest -q
```

See [ml_component/README.md](ml_component/README.md) for full documentation including directory layout and output descriptions.
