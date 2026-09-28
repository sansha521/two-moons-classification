# Two-moons comparison

## Implementation plan (implemented)

1. Generate 500 two-moons examples with Gaussian noise 0.20 for each seed 0–4.
2. Use stratified 60% train / 20% validation / 20% test partitions shared by all models. Fit StandardScaler on training data only; all models receive the same standardized coordinates.
3. Fit logistic regression (C=1, max_iter=2000), a perceptron (max_iter=2000, tol=0.001), a calibrated RBF SVM, an eight-unit MLP, and a PennyLane variational quantum classifier (VQC).
4. Fit sigmoid probability calibration for the frozen perceptron on validation data. Report calibrated classification metrics plus the original perceptron's accuracy. Calibration can move its decision boundary; these are distinct classifiers.
5. Fit an RBF SVM with C=1 and gamma='scale' on training data; freeze it and fit sigmoid calibration on validation data. Train an MLP with eight tanh hidden units using full-batch Adam, alpha=0.0001, learning rate 0.01, and 1000 epochs. Select the MLP checkpoint with the lowest validation log loss across epochs 1–1000.
6. Train the VQC for 100 full-batch Adam steps at learning rate 0.05. Use two qubits, three blocks of repeated Y-angle encoding, trainable Rot gates and CNOT, and 18 trainable parameters. Set p(class 1)=(1+<Z0>)/2. Minimize binary log loss. Select the checkpoint with the lowest validation log loss, including epoch zero.
7. Evaluate all models on the untouched test partition. Save per-run results and mean/sample standard deviation across seeds, data/split indices, package versions, and PNG figures.

No hyperparameter search is performed. Logistic regression uses its usual regularized logistic objective; the perceptron uses its own update rule. The common reported cost is unregularized binary log loss evaluated on predicted probabilities. Native optimization losses are not compared. Validation metrics are descriptive because that partition is used for calibration/checkpoint selection. Seeds vary both the generated dataset and training randomness, so standard deviations reflect both sources.

## Run

From this project directory, without activating the environment:

```bash
.venv/bin/python -m pip install -r requirements-compare.txt
.venv/bin/python compare.py
```

Quick check:

```bash
.venv/bin/python compare.py --samples 100 --epochs 3 --mlp-epochs 3 --seeds 0 --output /tmp/moons-smoke
```

Use `--help` for dataset size, noise, seeds, circuit layers, epochs, learning rate, and output directory. Existing files in the chosen output directory are overwritten. Plots use Agg and are saved to disk; no GUI or Tk is needed.

## Exact metrics

Labels are 0/1, class 1 is positive. Probability p is clipped to [1e-7, 1-1e-7]; predict class 1 when p >= 0.5.

| CSV field | Definition | Preferred direction |
|---|---|---|
| accuracy | (TP+TN)/N | Higher |
| balanced_accuracy | (TP/(TP+FN) + TN/(TN+FP))/2 | Higher |
| f1 | 2TP/(2TP+FP+FN), zero if denominator is zero | Higher |
| roc_auc | ROC area from probability scores | Higher |
| log_loss | -mean(y ln(p) + (1-y) ln(1-p)), natural logs | Lower |
| brier | mean((p-y)^2), binary range [0,1] | Lower |
| raw_accuracy | Accuracy of original uncalibrated perceptron; blank for other models | Higher |
| fit_seconds | Wall-clock model fitting, including calibration or quantum validation/checkpoint evaluation; excludes shared preprocessing | Lower |
| predict_ms_per_sample | Five warmed batch probability calls, total milliseconds / (5*N); excludes preprocessing and metric computation | Lower |

Accuracy and log loss are the primary metrics. Balanced accuracy, F1, ROC-AUC and Brier score give supporting views. Timing is machine-dependent and batch-amortized, not single-request latency. Quantum timings use exact expectation values on `default.qubit`, shots=None, with classical backpropagation. They do not measure quantum hardware or establish quantum advantage. Error bars are sample standard deviations, not confidence intervals; a single run has a blank SD.

## Output format

Default folder: `comparison_results/` next to the script.

- `metrics.csv`: one row per seed/model/split; fields:
  `seed,model,split,n,accuracy,balanced_accuracy,f1,roc_auc,log_loss,brier,raw_accuracy,fit_seconds,predict_ms_per_sample,selected_epoch`.
- `summary.csv`: test results only; fields `model,metric,mean,std,runs`.
- `quantum_history.csv`: `seed,epoch,train_log_loss,val_log_loss,train_accuracy,val_accuracy`. Epoch zero is initialization; only the VQC has epoch curves.
- `config.json`: command options, Python/platform and package versions. Fixed model settings are specified above and in the script.
- `data_seed_<seed>.npz`: original X/y, partition indices, fitted scaler mean/scale.
- `decision_boundaries_seed_<seed>.png`: all five selected classifiers, showing probabilities and held-out test points.
- `comparison.png`: test accuracy and log loss bars, mean ± sample SD.
- `quantum_learning_curves.png`: VQC train/validation cost and accuracy by epoch.

Console output gives seed, model, test accuracy, and test log loss for each completed fit. The calibrated perceptron is labeled `perceptron_calibrated`; its `raw_accuracy` column preserves the plain perceptron baseline.

## Additional baselines

The comparison includes two linear baselines (logistic regression and perceptron) and two nonlinear classical baselines (RBF SVM and a small MLP). Each has fixed settings; training budgets differ. Results describe this configuration and do not establish quantum advantage.

Possible future baselines include k-nearest neighbors, polynomial features plus logistic regression, and a class-prior DummyClassifier. Keep the same test partitions and define validation search budgets before tuning.

## References

- PennyLane angle encoding: https://docs.pennylane.ai/en/stable/code/api/pennylane.AngleEmbedding.html
- Probability calibration and frozen estimators: https://scikit-learn.org/stable/modules/calibration.html
- Classical classifiers on toy datasets: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
