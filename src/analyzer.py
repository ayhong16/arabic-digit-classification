import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity
from kmeans import k_means, plot_kmeans_contours, kmeans_component_gmm_helper
from em import em, plot_em_contours, em_component_gmm_helper
from util import compute_likelihood, get_comp_covariance, classify_utterance
from dataparser import DataParser
from gradientplotter import GradientPlotter


class Analyzer:

    def __init__(self):
        parser = DataParser()
        self.train_df = parser.train_df
        self.test_df = parser.test_df
        self.mfccs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        self.cov = {"tied": False, "cov_type": "full"}
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
            8: 4,
            9: 5,
        }

    def plot_timeseries(self, token, index, num_mfccs):
        metadata = self.get_single_training_utterance(token, index)
        data = metadata[1]
        x = metadata[0]
        y = []
        for i in range(num_mfccs):
            temp = []
            for j in range(len(x)):
                temp.append(data[j][i])
            y.append(temp)
        # plt.figure(1)  # comment out for making multiple graphs at once
        for k in range(len(y)):
            legend = 'MFCC ' + str(k + 1)
            plt.plot(x, y[k], label=legend)
        plt.legend()
        plt.xlabel('Analysis Window')
        plt.ylabel('MFCCs')
        plt.title('First ' + str(num_mfccs) + ' MFCCs For Single Utterance of ' + str(token))
        # plt.show()  # comment out for making multiple graphs at once

    def plot_scatter(self, token, index, comparisons):
        metadata = self.get_single_training_utterance(token, index)
        data = metadata[1]
        mapped_data = {}
        mfccs_needed = set()
        for comp1, comp2 in comparisons:
            mfccs_needed.add(comp1)
            mfccs_needed.add(comp2)
        for i in range(len(data)):
            temp = []
            if (i + 1) in mfccs_needed:
                for j in range(len(data)):
                    temp.append(data[j][i])
            mapped_data[i + 1] = temp
        # plt.figure(2)  # comment out for making multiple graphs at once
        for comp in comparisons:
            first_comp = comp[0]
            second_comp = comp[1]
            legend = "MFCC " + str(first_comp) + "(y) vs. MFCC " + str(second_comp) + "(x)"
            plt.scatter(mapped_data[second_comp], mapped_data[first_comp], label=legend)
        plt.legend()
        plt.title("Scatter Plot of Various MFCC Relationships in a Single Utterance of " + str(token))
        # plt.show()  # comment out for making multiple graphs at once

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

    def estimate_gmm_params(self, token):
        start = time.time()
        n_clusters = self.phoneme_map[token]
        tied = self.cov["tied"]
        covariance_type = self.cov["cov_type"]
        data = self.get_all_training_utterances(token)
        data = data[:, self.mfccs]
        if self.use_kmeans:
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
            "mfccs": self.mfccs
        }
        end = time.time()
        print(f"Time to train {token}: {end - start}")
        return ret

    def plot_likelihood_pdfs(self, gmm):
        fig, axs = plt.subplots(2, 5, tight_layout=True, sharex=True, sharey=True)

        for digit, ax in enumerate(axs.flatten()):
            filtered = self.train_df[self.train_df['Digit'] == digit]
            likelihoods = np.array(filtered['MFCCs'].apply(lambda x:
                                                           compute_likelihood(gmm, x))).reshape((-1, 1))
            kde = KernelDensity(bandwidth=40, kernel='gaussian')
            kde.fit(likelihoods)
            x_values = np.linspace(min(likelihoods), max(likelihoods), 1000).reshape(-1, 1)
            log_pdf = kde.score_samples(x_values)
            ax.plot(x_values, np.exp(log_pdf))
            ax.set_title(f"PDF for {digit}")
        plt.tight_layout()
        plt.suptitle(f"PDFs Using {gmm["covariance type"]} GMM for {gmm["digit"]}")
        plt.show()

    def classify_all_utterances(self, token, gmms):
        start = time.time()
        filtered = self.filter_test_utterances(token)
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

    def compute_confusion_matrix(self, tied, covariance_type, use_kmeans):
        confusion = np.zeros((10, 10))
        gmms = []
        for digit in range(10):
            gmms.append(self.estimate_gmm_params(digit))
        for digit in range(10):
            classification = self.classify_all_utterances(digit, gmms)
            confusion[digit, :] = classification
        print(confusion)

        fig, ax = plt.subplots()
        mapper = GradientPlotter((1, 0, 0), (0, 1, 0), 1)

        # Create a 2D list for cell colors
        cell_colors = [[mapper(confusion[i][j]) for j in range(len(confusion[i]))] for i in range(len(confusion))]
        ax.axis('off')
        ax.table(cellText=confusion, cellColours=cell_colors, loc='center', cellLoc='center', colLabels=[f"GMM {i}" for i in range(10)],
                 rowLabels=[f"Test {i}" for i in range(10)])
        ax.set_title(f"Confusion Matrix for "
                     f"{"Tied " if tied else "Distinct"} {covariance_type} "
                     f"Using {"KMeans" if use_kmeans else "EM"} GMMs")
        plt.show()

    def filter_test_utterances(self, token):
        return self.test_df[self.test_df['Digit'] == token]

    def get_all_training_utterances(self, token):
        filtered = self.train_df[self.train_df['Digit'] == token]
        data = []
        for index, row in filtered.iterrows():
            data.append(row['MFCCs'])
        return np.vstack(data)

    def get_single_training_utterance(self, token, index):
        filtered = self.train_df[self.train_df['Digit'] == token]
        filtered = filtered[filtered['Index'] == index]
        metadata = filtered.iloc[0]  # first token
        data = metadata['MFCCs']
        x = [i for i in range(len(data))]
        return x, data
