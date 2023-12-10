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
    comps = [(1, 2), (1, 6), (1, 13)]
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 14))
    for ax_num, ax in enumerate(axes.flatten()):
        for d in range(2, 3):
            if ax_num < 3:
                analyzer.plot_scatter(ax, d, [comps[ax_num]])
            else:
                analyzer.plot_em_gmm(d, [comps[ax_num - 3]], ax)
    plt.subplots_adjust(left=0.07, right=0.95, bottom=0.06, top=0.92, wspace=0.35, hspace=0.25)
    plt.legend(loc="best")
    plt.suptitle("Various 2D Cross Sections of all MFCCs for Digit 2")
    plt.savefig("../plots/mfcc_em_cross_sections.png")
    plt.show()


def plot_kmeans_contours():
    analyzer = Analyzer()
    comps = [(1, 2), (2, 3), (1, 3)]
    for d in range(10):
        analyzer.plot_kmeans_gmm(d, comps, analyzer.phoneme_map[d])
    plt.show()


def plot_likelihoods():
    analyzer = Analyzer()
    for d in range(1):
        gmm_params = analyzer.estimate_gmm_params(d)
        analyzer.plot_likelihood_pdfs(gmm_params)
        plt.savefig(f"../plots/likelihoods_digit_{d}.png")
        plt.show()


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


def create_cov_performance_map(analyzer, gender=None):
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
                confusion = analyzer.compute_confusion_matrix(cov_type=cov_type, use_kmeans=use_kmeans,
                                                              tied=tied, gender=gender)
                avg_accuracy = np.mean(compute_accuracy(confusion))
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
    accuracy_map, precision_map = create_cov_performance_map(analyzer, 'M')
    # overall
    # accuracy_map = {'diag': {'em': {'distinct': 85.77, 'tied': 82.46},
    #                          'kmeans': {'distinct': 52.86000000000001, 'tied': 74.07999999999998}},
    #                 'full': {'em': {'distinct': 90.23, 'tied': 88.36},
    #                          'kmeans': {'distinct': 87.53999999999999, 'tied': 87.91}},
    #                 'spherical': {'em': {'distinct': 73.27, 'tied': 74.09},
    #                               'kmeans': {'distinct': 52.86000000000001, 'tied': 74.07999999999998}}}
    # precision_map = {'diag': {'em': {'distinct': 0.8765173785778824, 'tied': 0.8443208527942513},
    #                           'kmeans': {'distinct': 0.6456228540617243, 'tied': 0.7766820352299685}},
    #                  'full': {'em': {'distinct': 0.9050832188255169, 'tied': 0.8887451716946844},
    #                           'kmeans': {'distinct': 0.8887162531107379, 'tied': 0.8878313045295714}},
    #                  'spherical': {'em': {'distinct': 0.7675264417271108, 'tied': 0.7756454945695905},
    #                                'kmeans': {'distinct': 0.6456228540617243, 'tied': 0.7766820352299685}}}
    # female
    # accuracy_map = {'diag': {'em': {'distinct': 93.27000000000001, 'tied': 92.18},
    #                          'kmeans': {'distinct': 55.010000000000005, 'tied': 82.80999999999999}},
    #                 'full': {'em': {'distinct': 90.64, 'tied': 92.1},
    #                          'kmeans': {'distinct': 90.83, 'tied': 90.1}},
    #                 'spherical': {'em': {'distinct': 82.53999999999999, 'tied': 82.55},
    #                               'kmeans': {'distinct': 55.010000000000005, 'tied': 82.80999999999999}}}
    # precision_map = {'diag': {'em': {'distinct': 0.9374554562218915, 'tied': 0.9258285472357984},
    #                           'kmeans': {'distinct': 0.5460513670668109,
    #                                      'tied': 0.8446610273864008}},
    #                  'full': {'em': {'distinct': 0.9082815586718807, 'tied': 0.9227000608298379},
    #                           'kmeans': {'distinct': 0.9132011460643265, 'tied': 0.9047112727998743}},
    #                  'spherical': {'em': {'distinct': 0.8505384553471439, 'tied': 0.8491389141874917},
    #                                'kmeans': {'distinct': 0.5460513670668109, 'tied': 0.8446610273864008}}}
    # male
    accuracy_map = {'diag': {'em': {'distinct': 85.01, 'tied': 87.0},
                             'kmeans': {'distinct': 60.370000000000005, 'tied': 69.83}},
                    'full': {'em': {'distinct': 86.44, 'tied': 87.91000000000001},
                             'kmeans': {'distinct': 85.64, 'tied': 87.09}},
                    'spherical': {'em': {'distinct': 72.09, 'tied': 71.44999999999999},
                                  'kmeans': {'distinct': 60.370000000000005, 'tied': 69.83}}}
    precision_map = {'diag': {'em': {'distinct': 0.8651645502007088, 'tied': 0.8878466433582464},
                              'kmeans': {'distinct': 0.6824426990843083, 'tied': 0.7963014734285093}},
                     'full': {'em': {'distinct': 0.8737526496427288, 'tied': 0.897020907490759},
                              'kmeans': {'distinct': 0.8789974437870782, 'tied': 0.885845190107872}},
                     'spherical': {'em': {'distinct': 0.7858736054257143, 'tied': 0.7813587584241425},
                                   'kmeans': {'distinct': 0.6824426990843083, 'tied': 0.7963014734285093}}}
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

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
    plot_bar_graph(categories, em_accuracy_vals, kmeans_accuracy_vals, axes[0], True)
    plot_bar_graph(categories, em_precision_vals, kmeans_precision_vals, axes[1], False)
    plt.legend(fontsize='large')
    plt.tight_layout()
    plt.savefig("../plots/male_cov_performance.png")
    plt.show()


def create_mfcc_accuracy_map(analyzer, gender=None):
    accuracy_map = {}
    mfccs = range(13)
    for n_keep in range(1, 14):
        accuracy_map[n_keep] = {
            "accuracy": 0,
            "mfccs": []
        }
    best_mfccs = []
    for n_keep in range(1, 14):
        best_mfcc = None
        best_accuracy = 0
        for mfcc in mfccs:
            if mfcc not in best_mfccs:
                temp = best_mfccs.copy()
                temp.append(mfcc)
                confusion = analyzer.compute_confusion_matrix(mfccs=temp, gender=gender)
                accuracy = np.mean(compute_accuracy(confusion))
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_mfcc = mfcc
                pprint(f"Testing mfccs: {temp}")
                pprint(f"Resulting Accuracy: {accuracy}. Best Accuracy: {best_accuracy}")
        best_mfccs.append(best_mfcc)
        accuracy_map[n_keep]["accuracy"] = best_accuracy
        accuracy_map[n_keep]["mfccs"] = best_mfccs.copy()
        pprint(f"Updated Best MFCCs: {best_mfccs}")
        pprint(f"Updated Accuracy map: {accuracy_map}")
    return accuracy_map


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
    ax.set_title(f'Male Average {"Accuracy" if isAccuracy else "Precision"} With Varying Parameters')
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
    for d in range(1, 2):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        analyzer.cov["tied"] = True
        analyzer.plot_em_gmm(d, comps, axes[0])
        analyzer.cov["tied"] = False
        analyzer.plot_em_gmm(d, comps, axes[1])
        plt.suptitle(f"Digit {d} Comparison of Tied and Distinct Covariance Matrices")
        plt.tight_layout()
        plt.savefig(f"../plots/tied_vs_distinct_digit_{d}.png")
    plt.show()


def plot_all_cov_contours():
    analyzer = Analyzer()
    comps = [(1, 2)]
    for d in range(5, 6):
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
        analyzer.cov["cov_type"] = "full"
        analyzer.plot_em_gmm(d, comps, axes[0])
        analyzer.cov["cov_type"] = "diag"
        analyzer.plot_em_gmm(d, comps, axes[1])
        analyzer.cov["cov_type"] = "spherical"
        analyzer.plot_em_gmm(d, comps, axes[2])
        plt.suptitle(f"Digit {d} Comparison of Full, Diag, & Spherical Covariance Matrices")
        plt.tight_layout()
        plt.savefig(f"../plots/all_cov_comparison_digit_{d}.png")
    plt.show()


def plot_greedy_mfcc():
    analyzer = Analyzer()
    # accuracy_map = create_mfcc_accuracy_map(analyzer, 'M')
    # overall
    # accuracy_map = {1: {'accuracy': 38.69, 'mfccs': [4]},
    #                 2: {'accuracy': 64.19, 'mfccs': [4, 2]},
    #                 3: {'accuracy': 76.66000000000001, 'mfccs': [4, 2, 1]},
    #                 4: {'accuracy': 83.05000000000001, 'mfccs': [4, 2, 1, 7]},
    #                 5: {'accuracy': 86.76, 'mfccs': [4, 2, 1, 7, 5]},
    #                 6: {'accuracy': 88.66999999999999, 'mfccs': [4, 2, 1, 7, 5, 10]},
    #                 7: {'accuracy': 88.91, 'mfccs': [4, 2, 1, 7, 5, 10, 12]},
    #                 8: {'accuracy': 89.21000000000001, 'mfccs': [4, 2, 1, 7, 5, 10, 12, 6]},
    #                 9: {'accuracy': 90.60000000000001, 'mfccs': [4, 2, 1, 7, 5, 10, 12, 6, 8]},
    #                 10: {'accuracy': 90.27, 'mfccs': [4, 2, 1, 7, 5, 10, 12, 6, 8, 9]},
    #                 11: {'accuracy': 89.72999999999999, 'mfccs': [4, 2, 1, 7, 5, 10, 12, 6, 8, 9, 11]},
    #                 12: {'accuracy': 90.27999999999999, 'mfccs': [4, 2, 1, 7, 5, 10, 12, 6, 8, 9, 11, 3]},
    #                 13: {'accuracy': 90.23, 'mfccs': [4, 2, 1, 7, 5, 10, 12, 6, 8, 9, 11, 3, 0]}}
    # female
    # accuracy_map = {1: {'accuracy': 58.36, 'mfccs': [4]},
    #                 2: {'accuracy': 82.09, 'mfccs': [4, 2]},
    #                 3: {'accuracy': 88.54, 'mfccs': [4, 2, 1]},
    #                 4: {'accuracy': 92.72999999999999, 'mfccs': [4, 2, 1, 7]},
    #                 5: {'accuracy': 93.19, 'mfccs': [4, 2, 1, 7, 10]},
    #                 6: {'accuracy': 93.17999999999998, 'mfccs': [4, 2, 1, 7, 10, 12]},
    #                 7: {'accuracy': 94.0, 'mfccs': [4, 2, 1, 7, 10, 12, 6]},
    #                 8: {'accuracy': 93.02000000000001, 'mfccs': [4, 2, 1, 7, 10, 12, 6, 11]},
    #                 9: {'accuracy': 92.36999999999998, 'mfccs': [4, 2, 1, 7, 10, 12, 6, 11, 5]},
    #                 10: {'accuracy': 91.72, 'mfccs': [4, 2, 1, 7, 10, 12, 6, 11, 5, 9]},
    #                 11: {'accuracy': 91.53999999999999, 'mfccs': [4, 2, 1, 7, 10, 12, 6, 11, 5, 9, 8]},
    #                 12: {'accuracy': 91.01, 'mfccs': [4, 2, 1, 7, 10, 12, 6, 11, 5, 9, 8, 0]},
    #                 13: {'accuracy': 90.64, 'mfccs': [4, 2, 1, 7, 10, 12, 6, 11, 5, 9, 8, 0, 3]}}
    # male
    accuracy_map = {1: {'accuracy': 34.63999999999999, 'mfccs': [10]},
                    2: {'accuracy': 57.720000000000006, 'mfccs': [10, 7]},
                    3: {'accuracy': 70.09, 'mfccs': [10, 7, 4]},
                    4: {'accuracy': 80.17999999999999, 'mfccs': [10, 7, 4, 6]},
                    5: {'accuracy': 84.28, 'mfccs': [10, 7, 4, 6, 3]},
                    6: {'accuracy': 87.47, 'mfccs': [10, 7, 4, 6, 3, 8]},
                    7: {'accuracy': 89.27, 'mfccs': [10, 7, 4, 6, 3, 8, 11]},
                    8: {'accuracy': 88.92, 'mfccs': [10, 7, 4, 6, 3, 8, 11, 2]},
                    9: {'accuracy': 88.55, 'mfccs': [10, 7, 4, 6, 3, 8, 11, 2, 9]},
                    10: {'accuracy': 87.91000000000001, 'mfccs': [10, 7, 4, 6, 3, 8, 11, 2, 9, 0]},
                    11: {'accuracy': 88.65, 'mfccs': [10, 7, 4, 6, 3, 8, 11, 2, 9, 0, 5]},
                    12: {'accuracy': 88.25999999999999, 'mfccs': [10, 7, 4, 6, 3, 8, 11, 2, 9, 0, 5, 1]},
                    13: {'accuracy': 86.44, 'mfccs': [10, 7, 4, 6, 3, 8, 11, 2, 9, 0, 5, 1, 12]}}
    num_mfccs = [i for i in range(1, 14)]
    accuracy_vals = []
    for n_keep in num_mfccs:
        accuracy_vals.append(accuracy_map[n_keep]["accuracy"])
    plt.figure(figsize=(10, 6))
    plt.plot(num_mfccs, accuracy_vals, marker='o')
    max_index = accuracy_vals.index(max(accuracy_vals))
    plt.scatter(num_mfccs[max_index], accuracy_vals[max_index], s=150, color='none', edgecolor='red',
                linewidth=2, zorder=5)
    plt.annotate(f'Max: {max(accuracy_vals):.2f}', (num_mfccs[max_index] + .1, accuracy_vals[max_index] - .5),
                 xytext=(num_mfccs[max_index] + 0.4, accuracy_vals[max_index] - 7),
                 arrowprops=dict(facecolor='black', arrowstyle='->'))
    plt.xlabel('Number of MFCCs Used')
    plt.ylabel('Accuracy (%)')
    plt.title('Male Evolution of Accuracy with Increasing Number of MFCCs')
    plt.grid(True)
    plt.savefig("../plots/male_greedy_mfcc_results.png")
    plt.show()


def create_cluster_effect_accuracy_map(analyzer):
    accuracy_map = {}
    clusters = [i for i in range(1, 9)]
    gmms = [analyzer.estimate_gmm_params(digit) for digit in range(10)]
    for d in range(10):
        accuracy_map[d] = []
    for i in clusters:
        analyzer.phoneme_map[0] = i
        print(f"Training with {analyzer.phoneme_map[0]} clusters.")
        gmms[0] = analyzer.estimate_gmm_params(0)
        confusion = analyzer.compute_confusion_matrix(gmms=gmms)
        accuracy = compute_accuracy(confusion)
        for d in range(10):
            accuracy_map[d].append(accuracy[d])
    return accuracy_map


def plot_cluster_effect():
    analyzer = Analyzer()
    # accuracy_map = create_cluster_effect_accuracy_map(analyzer)
    # Digit 0
    accuracy_map = {0: [18.6, 75.5, 79.10000000000001, 94.5, 96.39999999999999, 98.2, 97.3, 98.2],
                    1: [96.39999999999999, 96.39999999999999, 96.39999999999999, 96.39999999999999,
                        96.39999999999999, 96.39999999999999, 96.39999999999999, 96.39999999999999],
                    2: [81.8, 81.8, 81.8, 81.8, 81.8, 81.8, 81.8, 81.39999999999999],
                    3: [86.8, 86.4, 86.8, 85.0, 82.3, 82.3, 82.3, 80.0],
                    4: [86.8, 86.8, 86.8, 86.8, 86.4, 86.8, 86.4, 86.8],
                    5: [88.2, 88.2, 88.2, 88.2, 86.4, 85.9, 85.5, 85.0],
                    6: [96.39999999999999, 96.39999999999999, 96.39999999999999, 95.5, 92.7, 91.4, 90.9, 88.2],
                    7: [83.2, 83.2, 83.2, 82.69999999999999, 82.3, 81.8, 81.39999999999999, 80.5],
                    8: [95.89999999999999, 95.89999999999999, 95.89999999999999, 95.89999999999999, 95.89999999999999,
                        95.89999999999999, 95.89999999999999, 95.89999999999999],
                    9: [95.5, 95.5, 95.5, 95.5, 95.5, 95.5, 95.5, 95.5]}
    # Digit 9
    # accuracy_map = {0: [92.7, 92.7, 92.7, 94.5, 94.5, 94.5, 92.30000000000001, 90.5],
    #                 1: [96.39999999999999, 96.39999999999999, 96.39999999999999, 96.39999999999999, 96.39999999999999,
    #                     96.39999999999999, 96.39999999999999, 95.5],
    #                 2: [82.3, 82.3, 82.3, 82.3, 81.8, 81.39999999999999, 80.0, 79.5],
    #                 3: [85.0, 85.0, 85.0, 85.0, 85.0, 84.1, 79.10000000000001, 79.5],
    #                 4: [88.6, 88.6, 88.6, 88.6, 86.8, 85.5, 81.39999999999999, 80.5],
    #                 5: [88.6, 88.6, 88.6, 88.2, 88.2, 85.0, 82.69999999999999, 82.69999999999999],
    #                 6: [96.39999999999999, 96.39999999999999, 96.39999999999999, 96.39999999999999, 95.5, 88.2,
    #                     83.2, 85.0],
    #                 7: [86.8, 86.8, 86.8, 86.4, 82.69999999999999, 79.10000000000001, 74.5, 71.8],
    #                 8: [95.89999999999999, 95.89999999999999, 95.89999999999999, 95.89999999999999, 95.89999999999999,
    #                     95.0, 95.0, 95.0],
    #                 9: [21.4, 51.800000000000004, 69.5, 73.2, 95.5, 96.8, 97.3, 95.5]}
    pprint(accuracy_map)
    x = [i for i in range(1, 9)]
    for i in range(10):
        plt.plot(x, accuracy_map[i], marker='o', label=f"Digit {i} Accuracy")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Accuracy (%)")
    plt.title("Effect of Digit 0 Clusters on Accuracy for Each Digit")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../plots/global_cluster_effect_digit_0.png")
    plt.show()


def plot_gender_accuracy():
    analyzer = Analyzer()
    # overall = np.mean(compute_accuracy(analyzer.compute_confusion_matrix()))
    # female = np.mean(compute_accuracy(analyzer.compute_confusion_matrix(gender='F')))
    # male = np.mean(compute_accuracy(analyzer.compute_confusion_matrix(gender='M')))
    overall = 90.6
    female = 93.92
    male = 87.25999999999999
    print(f"Overall: {overall}, Female: {female}, Male: {male}")

    # Values and labels for the bar graph
    values = [overall, female, male]
    labels = ["Overall", "Female", "Male"]

    plt.bar(labels, values, color=['blue', 'orange', 'magenta'])
    plt.xlabel('Gender')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Gender')
    plt.ylim(50, 100)  # Set the y-axis limits between 0 and 1 for accuracy values
    plt.grid(axis='y')  # Show gridlines on the y-axis

    # Display the values on top of the bars
    for i, v in enumerate(values):
        plt.text(i, v + 0.15, f"{v:.2f}", ha='center')
    plt.tight_layout()
    plt.savefig("../plots/gender_accuracy.png")
    plt.show()


if __name__ == '__main__':
    # plot_timeseries()
    # plot_pairwise_scatter()
    # plot_kmeans_contours()
    plot_em_kmeans_contours()
    # plot_tied_distinct_contours()
    plot_likelihoods()
    # plot_em_contours()
    # plot_cov_performance()
    # plot_cluster_accuracy()
    # plot_greedy_mfcc()
    # plot_tied_distinct_contours()
    # plot_all_cov_contours()
    # plot_cluster_effect()
    # plot_gender_accuracy()
