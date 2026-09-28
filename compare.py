"""Reproducible two-moons benchmark. See COMPARISON.md for methodology."""
import argparse
import csv
from copy import deepcopy
import json
import platform
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_moons
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


MODEL_TITLES = {
    "logistic_regression": "Logistic regression",
    "perceptron_calibrated": "Perceptron (calibrated)",
    "rbf_svm_calibrated": "RBF SVM (calibrated)",
    "mlp": "MLP (8 hidden units)",
    "pennylane_vqc": "PennyLane VQC",
}


def split_data(y, seed):
    """Return stratified indices for 60% train, 20% validation, and 20% test."""
    train_idx, rest_idx = train_test_split(np.arange(len(y)), test_size=0.4, stratify=y, random_state=seed)
    val_idx, test_idx = train_test_split(rest_idx, test_size=0.5, stratify=y[rest_idx], random_state=seed)
    return train_idx, val_idx, test_idx


def metrics(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    pred = (p >= 0.5).astype(int)
    return dict(accuracy=accuracy_score(y, pred),
                balanced_accuracy=balanced_accuracy_score(y, pred),
                f1=f1_score(y, pred, zero_division=0), roc_auc=roc_auc_score(y, p),
                log_loss=float(-np.mean(y * np.log(p) + (1-y) * np.log(1-p))),
                brier=float(np.mean((p-y)**2)))


def fit_logistic_regression(X, y, seed):
    model = LogisticRegression(C=1.0, max_iter=2000, random_state=seed).fit(X, y)
    return lambda inputs: model.predict_proba(inputs)[:, 1]


def fit_perceptron(X, y, Xv, yv, seed):
    raw = Perceptron(max_iter=2000, tol=1e-3, random_state=seed).fit(X, y)
    model = CalibratedClassifierCV(FrozenEstimator(raw), method='sigmoid').fit(Xv, yv)
    return lambda inputs: model.predict_proba(inputs)[:, 1], raw


def fit_rbf_svm(X, y, Xv, yv, seed):
    raw = SVC(C=1.0, kernel='rbf', gamma='scale', random_state=seed).fit(X, y)
    model = CalibratedClassifierCV(FrozenEstimator(raw), method='sigmoid').fit(Xv, yv)
    return lambda inputs: model.predict_proba(inputs)[:, 1]


def fit_mlp(X, y, Xv, yv, seed, epochs, learning_rate):
    """Train one epoch at a time and keep the lowest validation log loss."""
    model = MLPClassifier(hidden_layer_sizes=(8,), activation='tanh', solver='adam',
                          alpha=0.0001, batch_size=len(y), learning_rate_init=learning_rate,
                          random_state=seed, shuffle=True)
    best_loss, best_epoch, best_model = float('inf'), 0, None
    for epoch in range(1, epochs + 1):
        model.partial_fit(X, y, classes=np.array([0, 1]))
        val_loss = metrics(yv, model.predict_proba(Xv)[:, 1])['log_loss']
        if val_loss < best_loss:
            best_loss, best_epoch, best_model = val_loss, epoch, deepcopy(model)
    return lambda inputs: best_model.predict_proba(inputs)[:, 1], best_epoch


def fit_quantum(X, y, Xv, yv, seed, epochs, layers, learning_rate):
    dev = qml.device('default.qubit', wires=2, shots=None)

    @qml.qnode(dev, interface='autograd', diff_method='backprop')
    def circuit(w, inputs):
        for block in w:
            qml.AngleEmbedding(inputs, wires=[0, 1], rotation='Y')
            for wire in range(2):
                qml.Rot(*block[wire], wires=wire)
            qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    def probability(w, inputs):
        return (1 + circuit(w, inputs)) / 2

    Xt = pnp.array(X, requires_grad=False)
    yt = pnp.array(y, requires_grad=False)

    def loss(w):
        p = pnp.clip(probability(w, Xt), 1e-7, 1-1e-7)
        return -pnp.mean(yt*pnp.log(p) + (1-yt)*pnp.log(1-p))

    w = pnp.array(np.random.default_rng(seed).normal(0, 0.1, (layers, 2, 3)), requires_grad=True)
    optimizer = qml.AdamOptimizer(learning_rate)
    best_loss, best_epoch, best_w = float('inf'), 0, w.copy()
    history = []
    for epoch in range(epochs + 1):
        if epoch:
            w = optimizer.step(loss, w)
        train = metrics(y, probability(w, X))
        val = metrics(yv, probability(w, Xv))
        history.append(dict(seed=seed, epoch=epoch, train_log_loss=train['log_loss'],
                            val_log_loss=val['log_loss'], train_accuracy=train['accuracy'],
                            val_accuracy=val['accuracy']))
        if val['log_loss'] < best_loss:
            best_loss, best_epoch, best_w = val['log_loss'], epoch, w.copy()
    return lambda inputs: np.asarray(probability(best_w, inputs)), best_epoch, history


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=500)
    parser.add_argument('--noise', type=float, default=0.2)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--mlp-epochs', type=int, default=1000)
    parser.add_argument('--mlp-learning-rate', type=float, default=0.01)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'comparison_results')
    args = parser.parse_args()
    if (args.mlp_epochs < 1 or args.mlp_learning_rate <= 0 or args.samples < 50 or args.noise < 0 or args.epochs < 1 or args.layers < 1
            or args.learning_rate <= 0 or len(set(args.seeds)) != len(args.seeds)
            or any(s < 0 or s >= 2**32 for s in args.seeds)):
        parser.error('Use samples >= 50, noise >= 0, positive epochs/layers/learning-rate, and unique seeds in [0, 2**32).')
    return args


def run_seed(args, seed):
    """Prepare data, fit each model, and evaluate one benchmark run."""
    rows, histories = [], []
    predictors = {}
    X, y = make_moons(n_samples=args.samples, noise=args.noise, random_state=seed)
    train_idx, val_idx, test_idx = split_data(y, seed)
    scaler = StandardScaler().fit(X[train_idx])
    Xt, Xv, Xe = [scaler.transform(X[idx]) for idx in (train_idx, val_idx, test_idx)]
    yt, yv, ye = [y[idx] for idx in (train_idx, val_idx, test_idx)]
    np.savez(args.output / f'data_seed_{seed}.npz', X=X, y=y, train_idx=train_idx,
             val_idx=val_idx, test_idx=test_idx, scaler_mean=scaler.mean_, scaler_scale=scaler.scale_)
    for name in MODEL_TITLES:
        start = perf_counter()
        raw = None
        best_epoch = ''
        if name == 'logistic_regression':
            predict = fit_logistic_regression(Xt, yt, seed)
        elif name == 'perceptron_calibrated':
            predict, raw = fit_perceptron(Xt, yt, Xv, yv, seed)
        elif name == 'rbf_svm_calibrated':
            predict = fit_rbf_svm(Xt, yt, Xv, yv, seed)
        elif name == 'mlp':
            predict, best_epoch = fit_mlp(Xt, yt, Xv, yv, seed, args.mlp_epochs, args.mlp_learning_rate)
        else:
            predict, best_epoch, history = fit_quantum(Xt, yt, Xv, yv, seed, args.epochs, args.layers, args.learning_rate)
            histories.extend(history)
        fit_seconds = perf_counter() - start
        predictors[name] = predict
        for split, inputs, labels in [('train', Xt, yt), ('validation', Xv, yv), ('test', Xe, ye)]:
            predict(inputs)  # warm up before timing
            start = perf_counter()
            for _ in range(5):
                probability = predict(inputs)
            predict_ms = (perf_counter() - start) * 1000 / (5 * len(labels))
            row = dict(seed=seed, model=name, split=split, n=len(labels), **metrics(labels, probability),
                       raw_accuracy=accuracy_score(labels, raw.predict(inputs)) if raw is not None else '',
                       fit_seconds=fit_seconds, predict_ms_per_sample=predict_ms, selected_epoch=best_epoch)
            rows.append(row)
            if split == 'test':
                print(f"seed={seed} {name:24s} accuracy={row['accuracy']:.3f} log_loss={row['log_loss']:.4f}", flush=True)
    plot_decision_boundaries(X, y, test_idx, scaler, predictors, seed, args.output)
    return rows, histories


def summarize_results(rows):
    summary = []
    names = list(dict.fromkeys(r['model'] for r in rows))
    for name in names:
        selected = [r for r in rows if r['model'] == name and r['split'] == 'test']
        for metric in ('accuracy', 'balanced_accuracy', 'f1', 'roc_auc', 'log_loss', 'brier',
                       'raw_accuracy', 'fit_seconds', 'predict_ms_per_sample'):
            values = [r[metric] for r in selected if r[metric] != '']
            if values:
                summary.append(dict(model=name, metric=metric, mean=float(np.mean(values)),
                                    std=float(np.std(values, ddof=1)) if len(values) > 1 else '', runs=len(values)))
    return summary


def save_results(args, rows, histories, summary):
    write_csv(args.output / 'metrics.csv', rows)
    write_csv(args.output / 'quantum_history.csv', histories)
    write_csv(args.output / 'summary.csv', summary)
    config = vars(args).copy()
    config['output'] = str(config['output'])
    config['python'] = platform.python_version()
    config['platform'] = platform.platform()
    config['versions'] = {p: version(p) for p in ('numpy', 'scikit-learn', 'pennylane', 'matplotlib')}
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')


def plot_decision_boundaries(X, y, test_idx, scaler, predictors, seed, output):
    """Plot fitted class probabilities and held-out points in original coordinates."""
    lower, upper = X.min(axis=0) - 0.3, X.max(axis=0) + 0.3
    xx, yy = np.meshgrid(np.linspace(lower[0], upper[0], 200),
                         np.linspace(lower[1], upper[1], 200))
    grid = scaler.transform(np.column_stack((xx.ravel(), yy.ravel())))
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True,
                             layout='constrained')
    axes = axes.ravel()
    for ax in axes[len(predictors):]:
        ax.set_visible(False)
    for ax, (name, predict) in zip(axes, predictors.items()):
        probabilities = np.concatenate([predict(batch) for batch in np.array_split(grid, 40)])
        surface = probabilities.reshape(xx.shape)
        shading = ax.contourf(xx, yy, surface, levels=np.linspace(0, 1, 21),
                              cmap='RdBu_r', vmin=0, vmax=1, alpha=0.75)
        if surface.min() < 0.5 < surface.max():
            ax.contour(xx, yy, surface, levels=[0.5], colors='black', linewidths=1.5)
        for label, color in [(0, '#2166ac'), (1, '#b2182b')]:
            points = X[test_idx[y[test_idx] == label]]
            ax.scatter(points[:, 0], points[:, 1], c=color, edgecolors='white',
                       linewidths=0.7, s=35, label=f'Test class {label}', zorder=3)
        accuracy = metrics(y[test_idx], predict(scaler.transform(X[test_idx])))['accuracy']
        ax.set(title=f'{MODEL_TITLES[name]}\nTest accuracy: {accuracy:.1%}',
               xlabel='Feature 1', xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]))
        ax.set_aspect('equal')
    axes[0].set_ylabel('Feature 2')
    axes[0].legend(loc='lower left', fontsize=8)
    fig.colorbar(shading, ax=list(axes[:len(predictors)]), shrink=0.65, label='Predicted probability of class 1')
    fig.suptitle(f'Two moons — seed {seed} | Black line: decision boundary (p = 0.5)')
    fig.savefig(output / f'decision_boundaries_seed_{seed}.png', dpi=180)
    plt.close(fig)


def plot_comparison(summary, output):
    names = list(dict.fromkeys(r['model'] for r in summary))
    labels = {'logistic_regression': 'Logistic', 'perceptron_calibrated': 'Perceptron\n(calibrated)',
              'rbf_svm_calibrated': 'RBF SVM\n(calibrated)', 'mlp': 'MLP\n(8 units)',
              'pennylane_vqc': 'PennyLane'}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, metric in zip(axes, ('accuracy', 'log_loss')):
        stats = [next(r for r in summary if r['model'] == name and r['metric'] == metric) for name in names]
        ax.bar([labels[name] for name in names], [r['mean'] for r in stats],
               yerr=[r['std'] or 0 for r in stats], capsize=4)
        ax.set_title(f'Test {metric} (mean ± sample SD)')
        if metric == 'accuracy':
            ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(output / 'comparison.png', dpi=150)
    plt.close(fig)


def plot_learning_curves(histories, seeds, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for seed in seeds:
        history = [r for r in histories if r['seed'] == seed]
        for ax, metric in zip(axes, ('log_loss', 'accuracy')):
            for split, style in [('train', '-'), ('val', '--')]:
                ax.plot([r['epoch'] for r in history], [r[f'{split}_{metric}'] for r in history],
                        style, label=f'{split}, seed {seed}')
            ax.set(xlabel='Epoch', ylabel=metric, title='PennyLane learning curves')
    axes[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(output / 'quantum_learning_curves.png', dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, histories = [], []
    for seed in args.seeds:
        seed_rows, seed_history = run_seed(args, seed)
        rows.extend(seed_rows)
        histories.extend(seed_history)

    summary = summarize_results(rows)
    save_results(args, rows, histories, summary)
    plot_comparison(summary, args.output)
    plot_learning_curves(histories, args.seeds, args.output)
    print(f'Results saved to {args.output}')


if __name__ == '__main__':
    main()
