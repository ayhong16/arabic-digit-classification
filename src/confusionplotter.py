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
    ax_matrix.set_ylabel("Predicted Values")
    ax_matrix.set_xlabel("True Values")

    for i in range(10):
        for j in range(10):
            ax_matrix.text(j, i, "{:.2f}".format(confusion_matrix[i, j]), ha='center', va='center', color='white')

    fig.colorbar(im, cax=ax_colorbar)
    plt.tight_layout()
    fig.savefig("../plots/stratified_confusion_matrix.png", dpi=300)
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


def plot_confusion_bar_graph(vals, ax, isAccuracy):
    categories = [i for i in range(10)]
    bar_width = 0.35

    # Create positions for each category
    x = np.arange(len(categories))

    # Plotting the grouped bar graph
    bars = ax.bar(x, vals, width=bar_width)  # Store the bars for reference

    if isAccuracy:
        y_ticks = np.linspace(50, 100, 6)
        ax.set_ylim(50, 100)
    else:
        y_ticks = np.linspace(0.5, 1.0, 6)
        ax.set_ylim(0.5, 1.0)
    ax.set_xlabel('Digit')
    ax.set_ylabel(f'{"Accuracy (%)" if isAccuracy else "Precision"}')
    ax.set_yticks(y_ticks)
    ax.yaxis.grid(True, zorder=1)
    ax.set_title(f'Per Digit {"Accuracy" if isAccuracy else "Precision"}')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)  # Set the x-axis labels

    # Annotate each bar with accuracy or precision values
    for i, v in enumerate(vals):
        ax.text(bars[i].get_x() + bars[i].get_width() / 2, v, f"{v:.2f}", ha='center', va='bottom')


def make_confusion_matrix():
    analyzer = Analyzer()
    confusion = analyzer.compute_confusion_matrix()
    accuracy = compute_accuracy(confusion)
    precision = compute_precision(confusion)
    print(f'Accuracy: {np.mean(accuracy)}')
    print(f'Precision: {np.mean(precision)}')
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
    plot_confusion_bar_graph(accuracy, axes[0], True)
    plot_confusion_bar_graph(precision, axes[1], False)
    plt.subplots_adjust(left=.08, right=.97, top=.93, bottom=.09, hspace=0.39, wspace=.2)
    fig.savefig("../plots/stratified_confusion_accuracy_precision.png", dpi=300)
    plot_confusion_matrix(confusion)


if __name__ == '__main__':
    make_confusion_matrix()
