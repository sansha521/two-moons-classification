# Two-moons classification: models and quantum encodings

Two related investigations use noisy two-moons data to compare classical and quantum classification, then examine how input encoding affects a PennyLane classifier.

## Investigations

| Investigation | Entry point | Methodology and results | Saved outputs |
| --- | --- | --- | --- |
| Classical vs. quantum models: logistic regression, perceptron, RBF SVM, MLP, and PennyLane VQC | [compare.py](compare.py) | [COMPARISON.md](COMPARISON.md) | [comparison_results/](comparison_results/) |
| Quantum input encoding: AngleEmbedding, BasisEmbedding, and AmplitudeEmbedding | [compare_encodings.py](compare_encodings.py) | [ENCODINGS.md](ENCODINGS.md) | [encoding_results/](encoding_results/) |

Both investigations use stratified 60% training, 20% validation, and 20% test splits, with preprocessing fitted on training data. Defaults are 500 samples, noise 0.2, and seeds 0–4. Each seed controls data generation, splitting, and model initialization.

The model comparison re-uploads angle inputs in every quantum circuit block. The encoding comparison embeds once before its trainable layers. See the investigation documents for these circuit choices and their effect on interpreting results.

## Development environment

The saved benchmark records the following environment in [config.json](comparison_results/config.json):

| Component | Recorded version |
|---|---|
| OS | Linux under WSL2, x86_64 |
| Kernel | 5.15.167.4-microsoft-standard-WSL2 |
| Python | 3.14.6 |
| NumPy | 2.5.3 |
| scikit-learn | 1.9.1 |
| PennyLane | 0.45.1 |
| Matplotlib | 3.11.2 |

Dependencies are pinned in [requirements-compare.txt](requirements-compare.txt). The quantum model uses PennyLane's `default.qubit` simulator with exact expectation values (`shots=None`). No quantum hardware is required. Matplotlib uses the `Agg` backend to save figures without a graphical display.

## Setup

From the project directory, create and activate a virtual environment using the Python version above to match the recorded environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-compare.txt
```

If `.venv` already exists, activate it and install the requirements.

## Run the investigations

Run commands from the repository root with the environment activated.

### Classical vs. quantum models

```bash
python compare.py
```

Saves metrics, data, and figures to `comparison_results/`. The default training budget is 100 steps for the VQC and 1,000 epochs for the MLP. See [COMPARISON.md](COMPARISON.md) for model settings, metric definitions, and results.

### Quantum encodings

```bash
python compare_encodings.py
```

Trains all three encodings for 100 steps each per seed and saves metrics, checkpoints, learning curves, classification diagrams, and test metric bar charts to `encoding_results/`. See [ENCODINGS.md](ENCODINGS.md) for preprocessing choices and results.

Regenerate encoding figures from saved checkpoints without retraining:

```bash
python compare_encodings.py --plots-only --output encoding_results
```

Both scripts support `--help` and `--output`. Existing files with matching names in the output directory are overwritten. Use a separate output directory to preserve the published results.

### Quick integration checks

```bash
python compare.py --samples 100 --epochs 3 --mlp-epochs 3 --seeds 0 --output /tmp/moons-smoke
python compare_encodings.py --samples 50 --epochs 3 --seeds 0 --output /tmp/encoding-smoke
```

These short runs check training, evaluation, and plot generation. They do not reproduce the full benchmark's performance.

## File architecture

```text
two-moons-classification/
├── README.md                      # Overview, setup, and entry points
├── requirements-compare.txt       # Shared pinned dependencies
├── .gitignore                     # Excludes local environments and caches
├── compare.py                     # Model comparison and shared utilities
├── COMPARISON.md                  # Model methodology, metrics, and results
├── comparison_results/
│   ├── config.json
│   ├── data_seed_<seed>.npz
│   ├── metrics.csv
│   ├── summary.csv
│   ├── quantum_history.csv
│   ├── comparison.png
│   ├── quantum_learning_curves.png
│   └── decision_boundaries_seed_<seed>.png
├── compare_encodings.py           # Quantum encoding comparison
├── ENCODINGS.md                   # Encoding methodology and results
└── encoding_results/
    ├── config.json
    ├── metrics.csv
    ├── summary.csv
    ├── training_history.csv
    ├── learning_curves.png
    ├── test_comparison.png
    ├── decision_boundaries_seed_<seed>.png
    ├── data_seed_<seed>.npz
    └── <Encoding>_seed_<seed>.npz
```

Keep both scripts at the root: `compare_encodings.py` imports splitting, evaluation, CSV, summary, and plotting utilities from `compare.py`. Markdown links and figures use paths relative to this root. Each results directory belongs to its corresponding investigation; the documents describe the saved runs and must be updated if those results are replaced.

The repository's `.gitignore` should exclude `.venv/`, `__pycache__/`, and `*.py[cod]`. The results folders contain the artifacts used by the investigation documents.
