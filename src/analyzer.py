import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity

from dataparser import DataParser
from em import em, plot_em_contours, em_component_gmm_helper, fix_em_cov_input
from kmeans import k_means, plot_kmeans_contours, kmeans_component_gmm_helper
from util import compute_likelihood, get_comp_covariance, classify_utterance


class Analyzer:

    def __init__(self):
        parser = DataParser()
        self.train_df = parser.train_df
        self.test_df = parser.test_df
        self.overall_mfccs = [1, 2, 4, 5, 6, 7, 8, 10, 12]
        # self.overall_mfccs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        self.cov = {"tied": False, "cov_type": "full"}
        self.female_mfccs = [1, 2, 4, 6, 7, 10, 12]
        self.male_mfccs = [3, 4, 6, 7, 8, 10, 11]
        self.use_kmeans = False
        self.phoneme_map = {
            0: 4,
            1: 4,
            2: 3,
            3: 4,
            4: 3,
            5: 3,
            6: 4,
            7: 4,
            8: 5,
            9: 5,
        }

    def plot_timeseries(self, token, index, num_mfccs, ax=None):
        metadata = self.get_single_training_utterance(token, index)
        data = metadata[1]
        x = metadata[0]
        y = []
        for i in range(num_mfccs):
            temp = []
            for j in range(len(x)):
                temp.append(data[j][i])
            y.append(temp)
        for k in range(len(y)):
            if ax is None:
                plt.plot(x, y[k], label=f"MFCC {k + 1}")
            else:
                ax.plot(x, y[k], label=f"MFCC {k + 1}")
        if ax is None:
            plt.xlabel('Analysis Window', fontsize='small')
            plt.ylabel('MFCCs', fontsize='small')
            plt.title(f'Digit {str(token)}', fontsize='small')
        else:
            ax.set_xlabel('Analysis Window', fontsize='small')
            ax.set_ylabel('MFCCs', fontsize='small')
            ax.set_title(f'Digit {str(token)}', fontsize='small')

    def plot_scatter(self, ax, token, comparisons):
        data = self.get_all_training_utterances(token)
        # metadata = self.get_single_training_utterance(token, 0)
        # data = data[:, self.mfccs]
        mapped_data = {}
        mfccs_needed = set()
        for comp1, comp2 in comparisons:
            mfccs_needed.add(comp1)
            mfccs_needed.add(comp2)
        for mfcc in mfccs_needed:
            mapped_data[mfcc] = data[:, mfcc - 1]
        # plt.figure(2)  # comment out for making multiple graphs at once
        for comp in comparisons:
            first_comp = comp[0]
            second_comp = comp[1]
            ax.scatter(mapped_data[second_comp], mapped_data[first_comp], label=f"Digit {token}", s=.1)
        ax.set_xlabel(f"MFCC {comparisons[0][1]}")
        ax.set_ylabel(f"MFCC {comparisons[0][0]}")

        # plt.show()  # comment out for making multiple graphs at once

    def plot_kmeans_gmm(self, token, comparisons, ax=None):
        data = self.get_all_training_utterances(token)
        # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
        colors = ["c", "r", "g", "m", 'y']
        cluster_info = k_means(self.phoneme_map[token], data)
        components = kmeans_component_gmm_helper(cluster_info, data, self.cov["cov_type"],
                                                 self.cov["tied"])
        i = 0
        temp = {}
        for key in cluster_info:
            temp[key] = {"data": cluster_info[key], "cov": components[i]["covariance"]}
            i += 1
        cluster_info = temp
        for i in range(len(comparisons)):
            first_comp = comparisons[i][0]
            second_comp = comparisons[i][1]
            color_ind = 0
            for key in cluster_info:
                c = colors[color_ind]
                coords = np.column_stack((cluster_info[key]["data"][:, second_comp - 1],
                                          cluster_info[key]["data"][:, first_comp - 1]))
                if ax is None:
                    plt.scatter(coords[:, 0], coords[:, 1], s=.1, color=c, alpha=.8)
                else:
                    ax.scatter(coords[:, 0], coords[:, 1], s=.1, color=c, alpha=.8)
                cov = get_comp_covariance(second_comp - 1, first_comp - 1, cluster_info[key]["cov"])
                plot_kmeans_contours(coords, cov, ax)
                color_ind += 1
            # ax.set_title("MFCC " + str(first_comp) + "(y) vs. MFCC" + str(second_comp) + "(x)")
            if ax is None:
                plt.title(f"K-Means Clustering")
                plt.xlabel("MFCC " + str(second_comp))
                plt.ylabel("MFCC " + str(first_comp))
                # plt.axis('equal')
            else:
                ax.set_title("K-Means Clustering")
                ax.set_xlabel("MFCC " + str(second_comp))
                ax.set_ylabel("MFCC " + str(first_comp))
                # ax.axis('equal')
        # fig.suptitle("K-Means GMMs for Various 2D Plots for Digit " + str(token))
        # plt.tight_layout()

    def plot_em_gmm(self, token, comparisons, ax=None):
        data = self.get_all_training_utterances(token)
        # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
        colors = ["c", "r", "g", "m", 'y']
        cov_in = fix_em_cov_input(self.cov["tied"], self.cov["cov_type"])
        n_components = self.phoneme_map[token]
        cluster_info = em(self.overall_mfccs, n_components, data, cov_in)
        for i in range(len(comparisons)):
            first_comp = comparisons[i][0]
            second_comp = comparisons[i][1]
            color_ind = 0
            for key in cluster_info:
                c = colors[color_ind]
                coords = np.column_stack(((cluster_info[key]["data"][:, second_comp - 1],
                                           cluster_info[key]["data"][:, first_comp - 1])))
                if ax is not None:
                    ax.scatter(coords[:, 0], coords[:, 1], s=.1, c=c, alpha=.8)
                else:
                    plt.scatter(coords[:, 0], coords[:, 1], s=.1, c=c, alpha=.8)
                cov = get_comp_covariance(second_comp - 1, first_comp - 1, cluster_info[key]["cov"])
                plot_em_contours([key[second_comp - 1], key[first_comp - 1]], cov, coords, ax)
                color_ind += 1
            # ax.set_title("MFCC " + str(first_comp) + "(y) vs. MFCC" + str(second_comp) + "(x)")
            if ax is not None:
                ax.set_title(
                    f"{"Tied" if self.cov["tied"] else "Distinct"} {self.cov["cov_type"]} Covariance")
                ax.set_xlabel("MFCC " + str(second_comp))
                ax.set_ylabel("MFCC " + str(first_comp))
                if self.cov["cov_type"] == "spherical":
                    ax.axis('equal')
            else:
                plt.title(f"EM Clustering")
                plt.xlabel("MFCC " + str(second_comp))
                plt.ylabel("MFCC " + str(first_comp))
                # plt.axis('equal')
        # fig.suptitle("EM GMMs for Various 2D Plots for Digit " + str(token))
        # plt.tight_layout()

    def estimate_gmm_params(self, token, cov_type=None, use_kmeans=None, tied=None, mfccs=None, gender=None):
        start = time.time()
        n_clusters = self.phoneme_map[token]
        if cov_type is None:
            cov_type = self.cov["cov_type"]
        if tied is None:
            tied = self.cov["tied"]
        if use_kmeans is None:
            use_kmeans = self.use_kmeans
        if mfccs is None:
            mfccs = self.overall_mfccs
        if gender == 'F':
            mfccs = self.female_mfccs
        if gender == 'M':
            mfccs = self.male_mfccs
        data = self.get_all_training_utterances(token, gender)
        data = data[:, mfccs]
        if use_kmeans:
            cluster_info = k_means(n_clusters, data)
            components = kmeans_component_gmm_helper(cluster_info, data, cov_type, tied)
        else:
            cov_in = fix_em_cov_input(tied, cov_type)
            cluster_info = em(mfccs, n_clusters, data, cov_type=cov_in)
            components = em_component_gmm_helper(cluster_info)
        cov_tied = "tied " if tied else "distinct "
        ret = {
            "digit": token,
            "number of components": n_clusters,
            "covariance type": cov_tied + cov_type,
            "components": components,
            "mfccs": mfccs
        }
        end = time.time()
        print(f"Time to train {token}: {end - start}")
        return ret

    def plot_likelihood_pdfs(self, gmm):
        fig, axs = plt.subplots(10, 1, figsize=(12, 8), tight_layout=True, sharex=True, sharey=True)
        for digit, ax in enumerate(axs.flatten()):
            filtered = self.train_df[self.train_df['Digit'] == digit]
            likelihoods = np.array(filtered['MFCCs'].apply(lambda x:
                                                           compute_likelihood(gmm, x))).reshape((-1, 1))
            kde = KernelDensity(bandwidth=40, kernel='gaussian')
            kde.fit(likelihoods)
            x_values = np.linspace(min(likelihoods), max(likelihoods), 1000).reshape(-1, 1)
            log_pdf = kde.score_samples(x_values)
            ax.plot(x_values, np.exp(log_pdf))
            plt.xlim((-1500, -200))
        plt.tight_layout()
        plt.suptitle(f"PDFs Using {gmm["covariance type"]} GMM for {gmm["digit"]}")
        plt.subplots_adjust(left=0.07, right=0.98, bottom=0.05, top=0.95, wspace=0.2, hspace=0.975)

    def classify_all_utterances(self, token, gmms, gender=None):
        start = time.time()
        filtered = self.filter_test_utterances(token, gender)
        classifications = np.zeros(10)
        mfccs = filtered["MFCCs"].to_numpy()
        for utterance in mfccs:
            classification = classify_utterance(utterance, gmms)
            classifications[classification] += 1
        total_utterances = len(mfccs)
        classifications = classifications / total_utterances
        end = time.time()
        print(f"Time to classify {token}: {end - start}")
        return np.round(classifications, 3)

    def compute_confusion_matrix(self, gmms=None, cov_type=None, use_kmeans=None, tied=None, mfccs=None, gender=None):
        confusion = np.zeros((10, 10))
        if cov_type is None:
            cov_type = self.cov["cov_type"]
        if use_kmeans is None:
            use_kmeans = self.use_kmeans
        if tied is None:
            tied = self.cov["tied"]
        if mfccs is None:
            mfccs = self.overall_mfccs
        # if gmms is None:
        #     gmms = [self.estimate_gmm_params(digit, cov_type=cov_type, use_kmeans=use_kmeans, tied=tied,
        #                                      mfccs=mfccs, gender=gender) for digit in range(10)]
        f_mfccs = self.female_mfccs
        m_mfccs = self.male_mfccs
        f_gmms = [self.estimate_gmm_params(digit, cov_type=cov_type, use_kmeans=use_kmeans, tied=tied,
                                           mfccs=f_mfccs, gender='F') for digit in range(10)]
        m_gmms = [self.estimate_gmm_params(digit, cov_type=cov_type, use_kmeans=use_kmeans, tied=tied,
                                           mfccs=m_mfccs, gender='M') for digit in range(10)]
        for digit in range(10):
            f_classification = self.classify_all_utterances(digit, f_gmms, 'F')
            m_classification = self.classify_all_utterances(digit, m_gmms, 'M')
            confusion[digit] = (f_classification + m_classification) / 2
            # confusion[digit] = m_classification
        return confusion

    def filter_test_utterances(self, token, gender=None):
        filtered = self.test_df[self.test_df['Digit'] == token]
        if gender is not None:
            filtered = filtered[filtered['Gender'] == gender]
        return filtered

    def get_all_training_utterances(self, token, gender=None):
        filtered = self.train_df[self.train_df['Digit'] == token]
        if gender is not None:
            filtered = filtered[filtered['Gender'] == gender]
        data = []
        for index, row in filtered.iterrows():
            data.append(row['MFCCs'])
        return np.vstack(data)

    def get_single_training_utterance(self, token, index):
        filtered = self.train_df[self.train_df['Digit'] == token]
        metadata = filtered.iloc[index]  # first token
        data = metadata['MFCCs']
        x = [i for i in range(len(data))]
        return x, np.array(data)
