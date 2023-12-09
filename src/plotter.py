from pprint import pprint

import numpy as np
from matplotlib import pyplot as plt
from analyzer import Analyzer
from confusionplotter import compute_accuracy, compute_precision, plot_confusion_matrix


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
    comps = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
             (8, 9), (9, 10), (10, 11), (11, 12), (12, 13)]
    fig, axes = plt.subplots(nrows=1, ncols=13)
    for ax in axes.flatten():
        for d in range(10):
            analyzer.plot_scatter(ax, d, comps)
    plt.legend()
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.06, top=0.92, wspace=0.35, hspace=0.25)
    plt.suptitle("2D Comparison of Various MFCCs")
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
    comps = [(1, 2)]
    analyzer.plot_em_gmm(0, comps)
    # comps = [(1, 2), (2, 3), (1, 3)]
    # for d in range(10):
    #     analyzer.plot_em_gmm(d, comps, analyzer.phoneme_map[d])
    plt.show()


def plot_em_kmeans_contours():
    analyzer = Analyzer()
    comps = [(1, 2)]
    for d in range(1):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        analyzer.plot_em_gmm(d, comps, axes[0])
        analyzer.plot_kmeans_gmm(d, comps, axes[1])
        plt.suptitle(f"Digit {d} Comparison of Tied Spherical EM and K-Means")
        plt.tight_layout()
        plt.axis('equal')
        plt.savefig(f"../plots/tied_spherical_kmeans_em_comparison_{d}.png")
    plt.show()


def create_cov_performance_map(analyzer):
    accuracy_map = {}
    precision_map = {}
    for cov_type in ["full", "diag", "spherical"]:
        accuracy_map[cov_type] = {}
        precision_map[cov_type] = {}
        for use_kmeans in [True, False]:
            label = kmeans_label(use_kmeans)
            accuracy_map[cov_type][label] = {}
            precision_map[cov_type][label] = {}
            for tied in [True, False]:
                accuracy_map[cov_type][label][tied_label(tied)] = 0
                precision_map[cov_type][label][tied_label(tied)] = 0
    for cov_type in ["full", "diag", "spherical"]:
        for use_kmeans in [True, False]:
            for tied in [True, False]:
                confusion = analyzer.compute_confusion_matrix(cov_type, use_kmeans, tied)
                avg_accuracy = np.mean(compute_accuracy(confusion)) * 100
                avg_precision = np.mean(compute_precision(confusion))
                accuracy_map[cov_type][kmeans_label(use_kmeans)][tied_label(tied)] = avg_accuracy
                precision_map[cov_type][kmeans_label(use_kmeans)][tied_label(tied)] = avg_precision
    return accuracy_map, precision_map


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
            accuracy = analyzer.classify_all_utterances(d, gmms)[d] * 100
            accuracy_map[d].append(accuracy)
    return accuracy_map


def plot_cov_performance():
    analyzer = Analyzer()
    # accuracy_map, precision_map = create_cov_performance_map(analyzer)
    accuracy_map = {'diag': {'em': {'distinct': 85.77, 'tied': 82.46},
                             'kmeans': {'distinct': 52.86000000000001, 'tied': 74.07999999999998}},
                    'full': {'em': {'distinct': 90.23, 'tied': 88.36},
                             'kmeans': {'distinct': 87.53999999999999, 'tied': 87.91}},
                    'spherical': {'em': {'distinct': 73.27, 'tied': 74.09},
                                  'kmeans': {'distinct': 52.86000000000001, 'tied': 74.07999999999998}}}
    precision_map = {'diag': {'em': {'distinct': 0.8765173785778824, 'tied': 0.8443208527942513},
                              'kmeans': {'distinct': 0.6456228540617243, 'tied': 0.7766820352299685}},
                     'full': {'em': {'distinct': 0.9050832188255169, 'tied': 0.8887451716946844},
                              'kmeans': {'distinct': 0.8887162531107379, 'tied': 0.8878313045295714}},
                     'spherical': {'em': {'distinct': 0.7675264417271108, 'tied': 0.7756454945695905},
                                   'kmeans': {'distinct': 0.6456228540617243, 'tied': 0.7766820352299685}}}
    pprint(accuracy_map)
    pprint(precision_map)

    # plot bar graph
    categories = ["Distinct Full", "Tied Full", "Distinct Diagonal",
                  "Tied Diagonal", "Distinct Spherical", "Tied Spherical"]
    em_accuracy_vals = []
    em_precision_vals = []
    for cov_type in ["full", "diag", "spherical"]:
        for tied in [False, True]:
            em_accuracy_vals.append(accuracy_map[cov_type][kmeans_label(False)][tied_label(tied)])
            em_precision_vals.append(precision_map[cov_type][kmeans_label(False)][tied_label(tied)])

    kmeans_accuracy_vals = []
    kmeans_precision_vals = []
    for cov_type in ["full", "diag", "spherical"]:
        for tied in [False, True]:
            kmeans_accuracy_vals.append(accuracy_map[cov_type][kmeans_label(True)][tied_label(tied)])
            kmeans_precision_vals.append(precision_map[cov_type][kmeans_label(True)][tied_label(tied)])

    fig, axes = plt.subplots(nrows=2, ncols=1)
    plot_bar_graph(categories, em_accuracy_vals, kmeans_accuracy_vals, axes[0], True)
    plot_bar_graph(categories, em_precision_vals, kmeans_precision_vals, axes[1], False)
    plt.legend(fontsize='large')
    plt.tight_layout()
    plt.show()


def plot_bar_graph(categories, em_vals, kmeans_vals, ax, isAccuracy):
    bar_width = 0.35

    # Create positions for each category
    x = np.arange(len(categories))

    # Plotting the grouped bar graph
    ax.bar(x - bar_width / 2, em_vals, width=bar_width, label='EM')
    ax.bar(x + bar_width / 2, kmeans_vals, width=bar_width, label='K-Means')

    if isAccuracy:
        y_ticks = np.linspace(50, 100, 6)
        ax.set_ylim(50, 100)
    else:
        y_ticks = np.linspace(0.5, 1.0, 6)
        ax.set_ylim(0.5, 1.0)
    ax.set_xlabel('Covariance Constraint')
    ax.set_ylabel(f'{"Accuracy (%)" if isAccuracy else "Precision"}')
    ax.set_yticks(y_ticks)
    ax.grid(True, zorder=1)
    ax.set_title(f'Average {"Accuracy" if isAccuracy else "Precision"} With Varying Parameters')
    ax.set_xticks(x, categories)


def plot_cluster_accuracy():
    analyzer = Analyzer()
    # accuracy_map = create_cluster_accuracy_map(analyzer)
    accuracy_map = {0: [18.6, 75.5, 79.10000000000001, 94.5, 96.39999999999999, 98.2, 97.3, 98.2],
                    1: [83.2, 86.8, 94.5, 96.39999999999999, 95.89999999999999, 97.3, 97.3, 97.3],
                    2: [12.7, 66.8, 81.39999999999999, 87.3, 85.9, 85.0, 86.4, 86.8],
                    3: [15.0, 50.0, 61.4, 78.60000000000001, 83.6, 89.5, 87.7, 88.2],
                    4: [5.8999999999999995, 61.8, 88.6, 91.8, 91.4, 89.5, 91.4, 93.2],
                    5: [53.6, 70.0, 85.0, 86.4, 91.8, 95.0, 94.5, 92.7],
                    6: [0.0, 50.9, 87.7, 86.8, 94.5, 95.89999999999999, 98.6, 98.6],
                    7: [6.800000000000001, 25.900000000000002, 30.0, 61.8, 74.5, 71.39999999999999, 73.6, 76.8],
                    8: [49.1, 78.60000000000001, 82.69999999999999, 88.2, 91.4, 90.5, 90.9, 89.5],
                    9: [18.2, 40.0, 55.50000000000001, 59.099999999999994, 80.9, 89.5, 92.7, 91.8]}
    pprint(accuracy_map)
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(16, 14))
    x = [i for i in range(1, 9)]
    for d, ax in enumerate(axes.flatten()):
        ax.plot(x, accuracy_map[d], marker='o')
        ax.set_xlabel("Number of Clusters", fontsize='small')
        ax.set_ylabel("Accuracy (%)", fontsize='small')
        ax.set_title(f"Digit {d}", fontsize='small')
        ax.grid(True)
    plt.suptitle("Influence of Number of Clusters on Accuracy for Each Digit")
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.06, top=0.92, wspace=0.35, hspace=0.25)
    # plt.tight_layout()
    plt.show()


def plot_tied_distinct_contours():
    analyzer = Analyzer()
    comps = [(1, 2)]
    for d in range(10):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        analyzer.cov["tied"] = True
        analyzer.plot_em_gmm(d, comps, axes[0])
        analyzer.cov["tied"] = False
        analyzer.plot_em_gmm(d, comps, axes[1], )
        plt.suptitle(f"Digit {d} Comparison of Tied and Distinct Covariance Matrices")
        # plt.savefig(f"../plots/tied_vs_distinct/digit_{d}.png")
        plt.tight_layout()
    plt.show()


def plot_greedy_mfcc():
    pass


if __name__ == '__main__':
    # plot_timeseries()
    # plot_pairwise_scatter()
    # plot_kmeans_contours()
    plot_em_kmeans_contours()
    # plot_tied_distinct_contours()
    # plot_likelihoods()
    # plot_em_contours()
    # plot_cov_performance()
    # plot_cluster_accuracy()
    # plot_greedy_mfcc()
