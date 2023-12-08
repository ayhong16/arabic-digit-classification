import numpy as np
from matplotlib import pyplot as plt

from analyzer import Analyzer
from gradientplotter import GradientPlotter


def plot_confusion_matrix(confusion):
    fig, ax = plt.subplots(figsize=(8, 6))

    mapper = GradientPlotter((1, 0, 0), (0, 1, 0), 1)

    # Create a 2D list for cell colors
    cell_colors = [[mapper(confusion[i][j]) for j in range(len(confusion[i]))] for i in range(len(confusion))]
    ax.axis('off')
    ax.table(cellText=confusion, cellColours=cell_colors, loc='center', cellLoc='center',
             colLabels=[f"GMM {i}" for i in range(10)],
             rowLabels=[f"Test {i}" for i in range(10)])
    ax.set_title("Confusion Matrix")
    plt.show()


def compute_accuracy(confusion):
    return np.diag(confusion)


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
