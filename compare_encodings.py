"""Compare PennyLane input encodings on two moons; see ENCODINGS.md."""
import argparse
import csv
import json
import platform
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

from compare import metrics, split_data, write_csv, summarize_results, plt

ENCODINGS = ('AngleEmbedding', 'BasisEmbedding', 'AmplitudeEmbedding')


def encode_inputs(name, X, thresholds):
    """Transform standardized coordinates using training-only statistics."""
    if name == 'AngleEmbedding':
        return np.asarray(X)
    if name == 'BasisEmbedding':
        return (X >= thresholds).astype(int)
    if name == 'AmplitudeEmbedding':
        # A constant reference retains radius and distinguishes opposite vectors.
        features = np.column_stack((X, np.ones(len(X)), np.zeros(len(X))))
        return features / np.linalg.norm(features, axis=1, keepdims=True)
    raise ValueError(f'Unknown encoding: {name}')


def make_probability(name):
    dev = qml.device('default.qubit', wires=2, shots=None)

    @qml.qnode(dev, interface='autograd', diff_method='backprop')
    def circuit(weights, inputs):
        if name == 'AngleEmbedding':
            qml.AngleEmbedding(inputs, wires=[0, 1], rotation='Y')
        elif name == 'BasisEmbedding':
            qml.BasisEmbedding(inputs, wires=[0, 1])
        elif name == 'AmplitudeEmbedding':
            qml.AmplitudeEmbedding(inputs, wires=[0, 1], normalize=False)
        else:
            raise ValueError(f'Unknown encoding: {name}')
        for block in weights:
            for wire in range(2):
                qml.Rot(*block[wire], wires=wire)
            qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    def probability(weights, inputs):
        if name == 'BasisEmbedding':
            # BasisEmbedding does not broadcast batches. Evaluate its four
            # possible states once and gather each example's probability.
            states = ((0, 0), (0, 1), (1, 0), (1, 1))
            values = pnp.stack([(1 + circuit(weights, state)) / 2 for state in states])
            indices = np.asarray(inputs, dtype=int) @ np.array([2, 1])
            return values[indices]
        return (1 + circuit(weights, inputs)) / 2

    return probability


def fit_encoding(name, Xt, yt, Xv, yv, initial_weights, args, seed):
    probability = make_probability(name)
    weights = pnp.array(initial_weights.copy(), requires_grad=True)
    inputs = pnp.array(Xt, requires_grad=False)
    labels = pnp.array(yt, requires_grad=False)

    def loss(w):
        p = pnp.clip(probability(w, inputs), 1e-7, 1 - 1e-7)
        return -pnp.mean(labels * pnp.log(p) + (1 - labels) * pnp.log(1 - p))

    optimizer = qml.AdamOptimizer(args.learning_rate)
    best_loss, best_epoch, best_weights = float('inf'), 0, weights.copy()
    history = []
    start = perf_counter()
    for epoch in range(args.epochs + 1):
        if epoch:
            weights = optimizer.step(loss, weights)
        train = metrics(yt, probability(weights, Xt))
        val = metrics(yv, probability(weights, Xv))
        history.append(dict(seed=seed, encoding=name, epoch=epoch,
                            train_log_loss=train['log_loss'], val_log_loss=val['log_loss'],
                            train_accuracy=train['accuracy'], val_accuracy=val['accuracy'],
                            elapsed_seconds=perf_counter() - start))
        if val['log_loss'] < best_loss:
            best_loss, best_epoch, best_weights = val['log_loss'], epoch, weights.copy()
    return lambda X: np.asarray(probability(best_weights, X)), best_weights, best_epoch, history


def plot_history(history, output):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for column, name in enumerate(ENCODINGS):
        for seed in sorted({r['seed'] for r in history}):
            selected = [r for r in history if r['encoding'] == name and r['seed'] == seed]
            for row, metric in enumerate(('log_loss', 'accuracy')):
                ax = axes[row, column]
                for split, style in (('train', '-'), ('val', '--')):
                    ax.plot([r['epoch'] for r in selected], [r[f'{split}_{metric}'] for r in selected],
                            style, label=f'{split}, seed {seed}')
                ax.set(xlabel='Epoch', ylabel=metric, title=name)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / 'learning_curves.png', dpi=150)
    plt.close(fig)



def plot_saved_results(output):
    """Render plots from saved checkpoints, using their original preprocessing."""
    config = json.loads((output / 'config.json').read_text())
    with (output / 'summary.csv').open() as handle:
        summary = list(csv.DictReader(handle))
    with (output / 'training_history.csv').open() as handle:
        history = [{k: (v if k == 'encoding' else float(v)) for k, v in r.items()}
                   for r in csv.DictReader(handle)]
    plot_history(history, output)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    colors = ['#4477AA', '#EEAA33', '#228833']
    for ax, metric, title in zip(axes, ('accuracy', 'log_loss'),
                                ('Test accuracy (higher is better)', 'Test log loss (lower is better)')):
        stats = [next(r for r in summary if r['model'] == name and r['metric'] == metric)
                 for name in ENCODINGS]
        means = [float(r['mean']) for r in stats]
        errors = [float(r['std']) if r['std'] else 0 for r in stats]
        bars = ax.bar([n.replace('Embedding', '') for n in ENCODINGS], means,
                      yerr=errors, capsize=5, color=colors, alpha=0.85)
        ax.bar_label(bars, labels=[f'{m:.3f}' for m in means], padding=35)
        ax.set(title=title, ylabel=metric)
        ax.set_ylim(0, 1.12 if metric == 'accuracy' else max(m + e for m, e in zip(means, errors)) * 1.35)
    fig.suptitle(f"Selected checkpoints: mean ± sample SD across {len(config['seeds'])} seeds")
    fig.savefig(output / 'test_comparison.png', dpi=170)
    plt.close(fig)

    for seed in config['seeds']:
        with np.load(output / f'data_seed_{seed}.npz') as data:
            X, y, test_idx = data['X'], data['y'], data['test_idx']
            lower, upper = X.min(axis=0) - 0.3, X.max(axis=0) + 0.3
            xx, yy = np.meshgrid(np.linspace(lower[0], upper[0], 200),
                                 np.linspace(lower[1], upper[1], 200))
            grid = np.column_stack((xx.ravel(), yy.ravel()))
            grid = (grid - data['scaler_mean']) / data['scaler_scale']
            test = (X[test_idx] - data['scaler_mean']) / data['scaler_scale']
            fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True,
                                     layout='constrained')
            for ax, name in zip(axes, ENCODINGS):
                with np.load(output / f'{name}_seed_{seed}.npz') as checkpoint:
                    weights = checkpoint['weights']
                    epoch = int(checkpoint['selected_epoch'])
                probability = make_probability(name)
                encoded = encode_inputs(name, grid, data['basis_thresholds'])
                surface = np.concatenate([np.asarray(probability(weights, batch))
                                          for batch in np.array_split(encoded, 40)]).reshape(xx.shape)
                # Nearest shading preserves the discontinuous basis regions.
                shading = ax.pcolormesh(xx, yy, surface, shading='nearest',
                                       cmap='RdBu_r', vmin=0, vmax=1, rasterized=True)
                if surface.min() < 0.5 < surface.max():
                    ax.contour(xx, yy, surface, levels=[0.5], colors='black', linewidths=1.3)
                for label, color in ((0, '#2166ac'), (1, '#b2182b')):
                    points = X[test_idx[y[test_idx] == label]]
                    ax.scatter(points[:, 0], points[:, 1], c=color, edgecolors='white',
                               linewidths=0.7, s=28, label=f'Test class {label}')
                scores = metrics(y[test_idx], probability(weights,
                                 encode_inputs(name, test, data['basis_thresholds'])))
                ax.set(title=f"{name} · epoch {epoch}\nAccuracy {scores['accuracy']:.1%} · log loss {scores['log_loss']:.3f}",
                       xlabel='Feature 1', xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]))
                ax.set_aspect('equal')
            axes[0].set_ylabel('Feature 2')
            axes[0].legend(loc='lower left', fontsize=8)
            fig.colorbar(shading, ax=list(axes), shrink=0.7, label='Predicted probability of class 1')
            fig.suptitle(f'Seed {seed} · validation-selected checkpoints · black line: p = 0.5')
            fig.savefig(output / f'decision_boundaries_seed_{seed}.png', dpi=170)
            plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plots-only', action='store_true', help='Regenerate figures from saved results in --output.')
    parser.add_argument('--samples', type=int, default=500)
    parser.add_argument('--noise', type=float, default=0.2)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'encoding_results')
    args = parser.parse_args()
    if (args.samples < 50 or not np.isfinite(args.noise) or args.noise < 0
            or args.epochs < 1 or args.layers < 1
            or not np.isfinite(args.learning_rate) or args.learning_rate <= 0
            or len(set(args.seeds)) != len(args.seeds)
            or any(s < 0 or s >= 2**32 for s in args.seeds)):
        parser.error('Use samples >= 50, finite noise >= 0, positive epochs/layers/learning-rate, '
                     'and unique seeds in [0, 2**32).')
    return args


def main():
    args = parse_args()
    if args.plots_only:
        plot_saved_results(args.output)
        print(f'Figures saved to {args.output}')
        return
    args.output.mkdir(parents=True, exist_ok=True)
    rows, histories = [], []
    for seed in args.seeds:
        X, y = make_moons(n_samples=args.samples, noise=args.noise, random_state=seed)
        indices = split_data(y, seed)
        scaler = StandardScaler().fit(X[indices[0]])
        standardized = scaler.transform(X)
        thresholds = np.median(standardized[indices[0]], axis=0)
        initial = np.random.default_rng(seed).normal(0, 0.1, (args.layers, 2, 3))
        np.savez(args.output / f'data_seed_{seed}.npz', X=X, y=y,
                 train_idx=indices[0], val_idx=indices[1], test_idx=indices[2],
                 scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
                 basis_thresholds=thresholds, initial_weights=initial)
        for name in ENCODINGS:
            encoded = encode_inputs(name, standardized, thresholds)
            Xt, Xv, Xe = [encoded[idx] for idx in indices]
            yt, yv, ye = [y[idx] for idx in indices]
            start = perf_counter()
            predict, weights, epoch, history = fit_encoding(name, Xt, yt, Xv, yv, initial, args, seed)
            fit_seconds = perf_counter() - start
            histories.extend(history)
            np.savez(args.output / f'{name}_seed_{seed}.npz', weights=np.asarray(weights),
                     selected_epoch=epoch, encoded_inputs=encoded)
            for split, inputs, labels in (('train', Xt, yt), ('validation', Xv, yv), ('test', Xe, ye)):
                row = dict(seed=seed, model=name, split=split, n=len(labels),
                           **metrics(labels, predict(inputs)), fit_seconds=fit_seconds, selected_epoch=epoch)
                rows.append(row)
                if split == 'test':
                    print(f"seed={seed} {name:20s} accuracy={row['accuracy']:.3f} "
                          f"log_loss={row['log_loss']:.4f} selected_epoch={epoch}", flush=True)
    # Shared summary helper expects these optional timing/baseline fields.
    summary = summarize_results([dict(r, raw_accuracy='', predict_ms_per_sample='') for r in rows])
    write_csv(args.output / 'metrics.csv', rows)
    write_csv(args.output / 'summary.csv', summary)
    write_csv(args.output / 'training_history.csv', histories)
    config = dict(vars(args), output=str(args.output), python=platform.python_version(),
                  versions={p: version(p) for p in ('pennylane', 'numpy', 'scikit-learn', 'matplotlib')},
                  encodings=list(ENCODINGS), qubits=2, embedding_repetitions=1,
                  angle_rotation='Y', basis_rule='feature >= training median',
                  amplitude_rule='L2-normalize [standardized x1, standardized x2, 1, 0]',
                  shots=None, optimizer='Adam', checkpoint='minimum validation log loss including epoch 0')
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    plot_saved_results(args.output)
    print(f'Results saved to {args.output}')


if __name__ == '__main__':
    main()
