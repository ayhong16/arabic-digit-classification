from pprint import pprint

import numpy as np
from matplotlib import pyplot as plt
from analyzer import Analyzer


def kmeans_label(kmeans):
    return f"{'kmeans' if kmeans else 'em'}"


def tied_label(use_tied):
    return f"{'tied' if use_tied else 'distinct'}"


def plot_timeseries():
    analyzer = Analyzer()
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(16, 14))
    for d, ax in enumerate(axes.flatten()):
        analyzer.plot_timeseries(d, 0, 13, ax)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.06, top=0.92, wspace=0.35, hspace=0.25)
    plt.suptitle("MFCC Evolution Throughout an Utterance of Each Digit")
    plt.show()


def plot_pairwise_scatter():
    analyzer = Analyzer()
    comps = [(1, 2), (2, 3), (1, 3)]
    for d in range(10):
        plt.figure(d)
        analyzer.plot_scatter(d, 1, comps)
    plt.show()


def plot_kmeans_contours():
    analyzer = Analyzer()
    comps = [(1, 2), (2, 3), (1, 3)]
    for d in range(10):
        analyzer.plot_kmeans_gmm(d, comps, analyzer.phoneme_map[d])
    plt.show()


def plot_likelihoods():
    analyzer = Analyzer()
    digit = 9
    gmm_params = analyzer.estimate_gmm_params(digit)
    analyzer.plot_likelihood_pdfs(gmm_params)


def plot_em_contours():
    analyzer = Analyzer()
    comps = [(1, 2), (2, 3), (1, 3)]
    for d in range(10):
        analyzer.plot_em_gmm(d, comps, analyzer.phoneme_map[d], "full")
    plt.show()


def create_cov_accuracy_map(analyzer):
    accuracy_map = {}
    for cov_type in ["full", "diag", "spherical"]:
        accuracy_map[cov_type] = {}
        for use_kmeans in [True, False]:
            label = kmeans_label(use_kmeans)
            accuracy_map[cov_type][label] = {}
            for tied in [True, False]:
                accuracy_map[cov_type][label][tied_label(tied)] = 0
    for cov_type in ["full", "diag", "spherical"]:
        for use_kmeans in [True, False]:
            for tied in [True, False]:
                confusion = analyzer.compute_confusion_matrix(cov_type, use_kmeans, tied)
                avg_accuracy = np.mean(np.diagonal(confusion))
                accuracy_map[cov_type][kmeans_label(use_kmeans)][tied_label(tied)] = avg_accuracy


def create_cluster_accuracy_map(analyzer):
    accuracy_map = {}
    clusters = [i for i in range(1, 9)]
    temp_gmms = [analyzer.estimate_gmm_params(digit) for digit in range(10)]
    for d in range(10):
        accuracy_map[d] = []
    for d in range(10):
        gmms = temp_gmms
        for i in clusters:
            analyzer.phoneme_map[d] = i
            print(f"Training digit {d} with {analyzer.phoneme_map[d]} clusters.")
            gmms[d] = analyzer.estimate_gmm_params(d)
            accuracy = analyzer.classify_all_utterances(d, gmms)[d]
            accuracy_map[d].append(accuracy)
    return accuracy_map


def plot_cov_accuracy():
    analyzer = Analyzer()
    # accuracy_map = create_cov_accuracy_map(analyzer)
    accuracy_map = {'diag': {'em': {'distinct': 0.8675999999999998, 'tied': 0.8544},
                             'kmeans': {'distinct': 0.5287, 'tied': 0.7313}},
                    'full': {'em': {'distinct': 0.8968999999999999, 'tied': 0.8953},
                             'kmeans': {'distinct': 0.8744, 'tied': 0.8795}},
                    'spherical': {'em': {'distinct': 0.7268, 'tied': 0.7268},
                                  'kmeans': {'distinct': 0.5287, 'tied': 0.7313}}}
    pprint(accuracy_map)

    # plot bar graph
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
    plt.show()


def plot_cluster_accuracy():
    analyzer = Analyzer()
    # accuracy_map = create_cluster_accuracy_map(analyzer)
    accuracy_map = {0: [0.186, 0.755, 0.895, 0.945, 0.964, 0.982, 0.973, 0.968],
                    1: [0.832, 0.868, 0.95, 0.968, 0.964, 0.977, 0.973, 0.977],
                    2: [0.141, 0.677, 0.818, 0.777, 0.859, 0.873, 0.877, 0.868],
                    3: [0.136, 0.514, 0.655, 0.795, 0.827, 0.909, 0.877, 0.8],
                    4: [0.059, 0.618, 0.877, 0.914, 0.909, 0.923, 0.936, 0.923],
                    5: [0.532, 0.718, 0.855, 0.877, 0.927, 0.959, 0.968, 0.968],
                    6: [0.0, 0.509, 0.859, 0.832, 0.859, 0.986, 0.964, 0.964],
                    7: [0.055, 0.277, 0.577, 0.632, 0.732, 0.732, 0.773, 0.805],
                    8: [0.523, 0.791, 0.841, 0.895, 0.909, 0.886, 0.873, 0.918],
                    9: [0.2, 0.427, 0.536, 0.741, 0.777, 0.891, 0.927, 0.923]}
    pprint(accuracy_map)
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(16, 14))
    x = [i for i in range(1, 9)]
    for d, ax in enumerate(axes.flatten()):
        ax.plot(x, accuracy_map[d], marker='o')
        ax.set_xlabel("Number of Clusters", fontsize='small')
        ax.set_ylabel("Accuracy", fontsize='small')
        ax.set_title(f"Digit {d}", fontsize='small')
        ax.grid(True)
    plt.suptitle("Influence of Number of Clusters on Accuracy for Each Digit")
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.06, top=0.92, wspace=0.35, hspace=0.25)
    # plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # plot_timeseries()
    plot_pairwise_scatter()
    # plot_kmeans_contours()
    # plot_likelihoods()
    # plot_em_contours()
    plot_cov_accuracy()
    # plot_cluster_accuracy()
