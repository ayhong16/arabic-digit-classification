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


def make_confusion_matrix():
    analyzer = Analyzer()
    confusion = analyzer.compute_confusion_matrix()
    print(np.mean(np.diagonal(confusion)))
    plot_confusion_matrix(confusion)


if __name__ == '__main__':
    make_confusion_matrix()
