from pprint import pprint

import numpy as np
from matplotlib import pyplot as plt
from gradientplotter import GradientPlotter
from analyzer import Analyzer


def kmeans_label(kmeans):
    return f"{'kmeans' if kmeans else 'em'}"


def tied_label(use_tied):
    return f"{'tied' if use_tied else 'distinct'}"


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


def plot_accuracy(accuracy_map):
    categories = ["Distinct Full", "Tied Full", "Distinct Diagonal",
                  "Tied Diagonal", "Distinct Spherical", "Tied Spherical"]
    em_vals = []
    for cov_type in ["full", "diag", "spherical"]:
        for tied in [False, True]:
            em_vals.append(accuracy_map[cov_type][kmeans_label(False)][tied_label(tied)])

    kmeans_vals = []
    for cov_type in ["full", "diag", "spherical"]:
        for tied in [False, True]:
            kmeans_vals.append(accuracy_map[cov_type][kmeans_label(True)][tied_label(tied)])

    bar_width = 0.35

    # Create positions for each category
    x = np.arange(len(categories))

    # Plotting the grouped bar graph
    plt.bar(x - bar_width / 2, em_vals, width=bar_width, label='EM')
    plt.bar(x + bar_width / 2, kmeans_vals, width=bar_width, label='K-Means')

    y_ticks = np.linspace(0.5, 1.0, 6)
    plt.xlabel('Covariance Constraint')
    plt.ylabel('Accuracy')
    plt.ylim(0.5, 1.0)
    plt.yticks(y_ticks)
    plt.grid(True, zorder=1)
    plt.title('Average Accuracy With Varying Parameters')
    plt.xticks(x, categories)
    plt.legend(fontsize='large')

    plt.tight_layout()
    plt.show()


def plot_average_accuracy():
    analyzer = Analyzer()
    # accuracy_map = {}
    # for cov_type in ["full", "diag", "spherical"]:
    #     accuracy_map[cov_type] = {}
    #     for use_kmeans in [True, False]:
    #         label = kmeans_label(use_kmeans)
    #         accuracy_map[cov_type][label] = {}
    #         for tied in [True, False]:
    #             accuracy_map[cov_type][label][tied_label(tied)] = 0
    # for cov_type in ["full", "diag", "spherical"]:
    #     for use_kmeans in [True, False]:
    #         for tied in [True, False]:
    #             confusion = analyzer.compute_confusion_matrix(cov_type, use_kmeans, tied)
    #             avg_accuracy = np.mean(np.diagonal(confusion))
    #             accuracy_map[cov_type][kmeans_label(use_kmeans)][tied_label(tied)] = avg_accuracy
    accuracy_map = {'diag': {'em': {'distinct': 0.8675999999999998, 'tied': 0.8544},
                             'kmeans': {'distinct': 0.5287, 'tied': 0.7313}},
                    'full': {'em': {'distinct': 0.8968999999999999, 'tied': 0.8953},
                             'kmeans': {'distinct': 0.8744, 'tied': 0.8795}},
                    'spherical': {'em': {'distinct': 0.7268, 'tied': 0.7268},
                                  'kmeans': {'distinct': 0.5287, 'tied': 0.7313}}}
    pprint(accuracy_map)
    plot_accuracy(accuracy_map)
    plt.show()


def make_confusion_matrix():
    analyzer = Analyzer()
    confusion = analyzer.compute_confusion_matrix()
    plot_confusion_matrix(confusion)


if __name__ == '__main__':
    # make_confusion_matrix()
    plot_average_accuracy()
