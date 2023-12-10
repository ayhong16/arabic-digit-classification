from pprint import pprint

import numpy as np
from matplotlib import pyplot as plt, colormaps

from analyzer import Analyzer


def plot_confusion_matrix(confusion_matrix, title="Confusion Matrix"):
    fig, (ax_matrix, ax_colorbar) = plt.subplots(1, 2,
                                                 gridspec_kw={'width_ratios': [4, .1]},
                                                 figsize=(9, 8))

    viridis_cmap = colormaps.get_cmap('Spectral')
    im = ax_matrix.imshow(confusion_matrix, cmap=viridis_cmap)

    fig.suptitle(title)
    ax_matrix.set_xticks(np.arange(10))
    ax_matrix.set_yticks(np.arange(10))
    ax_matrix.set_xticklabels([str(i) for i in range(10)])
    ax_matrix.set_yticklabels([str(i) for i in range(10)])
    ax_matrix.set_xlabel("Predicted Values")
    ax_matrix.set_ylabel("True Values")

    for i in range(10):
        for j in range(10):
            ax_matrix.text(j, i, "{:.2f}".format(confusion_matrix[i, j]), ha='center', va='center', color='white')

    fig.colorbar(im, cax=ax_colorbar)
    plt.tight_layout()
    plt.show()


def compute_accuracy(confusion):
    return np.diag(confusion) * 100


def compute_precision(confusion):
    precisions = []
    for i in range(confusion.shape[0]):
        true_positives = confusion[i, i]
        false_positives = np.sum(confusion[:, i]) - true_positives
        precision = true_positives / (
                true_positives + false_positives + 1e-9)  # Adding a small value to avoid division by zero
        precisions.append(precision)
    return precisions


def make_confusion_matrix():
    analyzer = Analyzer()
    confusion = analyzer.compute_confusion_matrix()
    print(f'Accuracy: {np.mean(compute_accuracy(confusion))}')
    print(f'Precision: {np.mean(compute_precision(confusion))}')
    plot_confusion_matrix(confusion)


if __name__ == '__main__':
    make_confusion_matrix()
