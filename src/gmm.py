import time

import numpy as np
from matplotlib import pyplot as plt
from dataparser import DataParser

from em import em_component_gmm_helper, em, plot_em_contours
from kmeans import k_means, kmeans_component_gmm_helper, plot_kmeans_contours
from util import get_comp_covariance


class GMM:
    def __init__(self, param_map, train_df=None, test_df=None):
        self.param_map = param_map
        if train_df is None or test_df is None:
            parser = DataParser
        self.train_df = train_df
        self.test_df = test_df
        self.gmms = [self.estimate_gmm_params(token) for token in range(10)]

    def estimate_gmm_params(self, token):
        start = time.time()
        n_clusters = self.param_map[token]["num_phonemes"]
        tied = self.param_map[token]["tied"]
        covariance_type = self.param_map[token]["cov_type"]
        use_kmeans = self.param_map[token]["use_kmeans"]
        data = self.get_all_training_utterances(token)
        data = data[:, self.param_map[token]["mfccs"]]
        if use_kmeans:
            cluster_info = k_means(n_clusters, data)
            components = kmeans_component_gmm_helper(cluster_info, data, covariance_type, tied)
        else:
            cluster_info = em(n_clusters, data, covariance_type)
            components = em_component_gmm_helper(cluster_info)
        cov_tied = "tied " if tied else "distinct "
        ret = {
            "digit": token,
            "number of components": n_clusters,
            "covariance type": cov_tied + covariance_type,
            "components": components,
            "mfccs": self.param_map[token]["mfccs"]
        }
        end = time.time()
        print(f"Time to train {token}: {end - start}")
        return ret

    def plot_kmeans_gmm(self, token, comparisons, n_clusters):
        data = self.get_all_training_utterances(token)
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
        colors = ["c", "r", "g", "m", 'y']
        cluster_info = k_means(n_clusters, data)
        for i in range(len(comparisons)):
            first_comp = comparisons[i][0]
            second_comp = comparisons[i][1]
            color_ind = 0
            for key in cluster_info:
                c = colors[color_ind]
                coords = np.column_stack((cluster_info[key][:, second_comp - 1],
                                          cluster_info[key][:, first_comp - 1]))
                axes[i].scatter(coords[:, 0], coords[:, 1], s=.1, color=c, alpha=.8)
                plot_kmeans_contours(coords, axes[i], c)
                color_ind += 1
            axes[i].set_title("MFCC " + str(first_comp) + "(y) vs. MFCC" + str(second_comp) + "(x)")
        fig.suptitle("K-Means GMMs for Various 2D Plots for Digit " + str(token))
        plt.tight_layout()

    def plot_em_gmm(self, token, comparisons, n_components, cov_type):
        data = self.get_all_training_utterances(token)
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
        colors = ["c", "r", "g", "m", 'y']
        cluster_info = em(n_components, data, cov_type)
        for i in range(len(comparisons)):
            first_comp = comparisons[i][0]
            second_comp = comparisons[i][1]
            color_ind = 0
            for key in cluster_info:
                c = colors[color_ind]
                coords = np.column_stack(((cluster_info[key]["data"][:, second_comp - 1],
                                           cluster_info[key]["data"][:, first_comp - 1])))
                axes[i].scatter(coords[:, 0], coords[:, 1], s=.1, c=c, alpha=.8)
                cov = get_comp_covariance(second_comp - 1, first_comp - 1, cluster_info[key]["cov"])
                plot_em_contours([key[second_comp - 1], key[first_comp - 1]], cov, c, axes[i], coords)
                color_ind += 1
            axes[i].set_title("MFCC " + str(first_comp) + "(y) vs. MFCC" + str(second_comp) + "(x)")
        fig.suptitle("EM GMMs for Various 2D Plots for Digit " + str(token))
        plt.tight_layout()

    def get_all_training_utterances(self, token):
        filtered = self.train_df[self.train_df['Digit'] == token]
        data = []
        for index, row in filtered.iterrows():
            data.append(row['MFCCs'])
        return np.vstack(data)
