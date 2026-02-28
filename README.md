# ai-ml-component

A repository containing a reproducible machine-learning component for binary classification.

## ML Component

A reproducible machine-learning component is available in [`ml_component/`](ml_component/README.md).
See [ml_component/README.md](ml_component/README.md) for setup, training, evaluation, and test instructions.

---

## Development Note

Parts of this project were developed with the assistance of AI coding tools (such as GitHub Copilot).
If you are asked about your development process — for example, during a review or interview — please be
transparent and honest about which tools you used. Using AI assistants is a normal part of modern
software development, but misrepresenting your workflow is not acceptable.
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
