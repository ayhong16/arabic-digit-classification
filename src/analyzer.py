import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity
from kmeans import k_means, plot_kmeans_contours, kmeans_component_gmm_helper
from em import em, plot_em_contours, em_component_gmm_helper
from util import compute_likelihood, get_comp_covariance, classify_utterance
from dataparser import DataParser


class Analyzer:

    def __init__(self):
        train_parser = DataParser()
        train_parser.parse_txt("Train_Arabic_Digit.txt", 66)
        self.train_df = train_parser.get_dataframe()
        test_parser = DataParser()
        test_parser.parse_txt("Test_Arabic_Digit.txt", 22)
        self.test_df = test_parser.get_dataframe()
        self.param_map = {
            0: {"num_phonemes": 4,
                "mfccs": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                "tied": False,
                "cov_type": "full",
                "use_kmeans": False},
            1: {"num_phonemes": 4,
                "mfccs": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                "tied": False,
                "cov_type": "full",
                "use_kmeans": False},
            2: {"num_phonemes": 3,
                "mfccs": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                "tied": False,
                "cov_type": "full",
                "use_kmeans": False},
            3: {"num_phonemes": 4,
                "mfccs": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                "tied": False,
                "cov_type": "full",
                "use_kmeans": False},
            4: {"num_phonemes": 3,
                "mfccs": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                "tied": False,
                "cov_type": "full",
                "use_kmeans": False},
            5: {"num_phonemes": 3,
                "mfccs": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                "tied": False,
                "cov_type": "full",
                "use_kmeans": False},
            6: {"num_phonemes": 4,
                "mfccs": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                "tied": False,
                "cov_type": "full",
                "use_kmeans": False},
            7: {"num_phonemes": 3,
                "mfccs": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                "tied": False,
                "cov_type": "full",
                "use_kmeans": False},
            8: {"num_phonemes": 4,
                "mfccs": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                "tied": False,
                "cov_type": "full",
                "use_kmeans": False},
            9: {"num_phonemes": 3,
                "mfccs": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                "tied": False,
                "cov_type": "full",
                "use_kmeans": False},
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

    def estimate_gmm_params(self, token, selected_mfccs):
        n_clusters = self.param_map[token]["num_phonemes"]
        tied = self.param_map[token]["tied"]
        covariance_type = self.param_map[token]["cov_type"]
        use_kmeans = self.param_map[token]["use_kmeans"]
        data = self.get_all_training_utterances(token)
        data = data[:, selected_mfccs]
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
            "mfccs": selected_mfccs
        }
        return ret

    def plot_likelihood_pdfs(self, gmm):
        fig, axs = plt.subplots(2, 5, tight_layout=True, sharex=True, sharey=True)

        for digit, ax in enumerate(axs.flatten()):
            filtered = self.train_df[self.train_df['Digit'] == digit]
            selected = [0, 7, 9, 12]
            likelihoods = np.array(filtered['MFCCs'].apply(lambda x:
                                                           compute_likelihood(gmm, x, selected))).reshape((-1, 1))
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
        filtered = self.filter_test_utterances(token)
        classifications = np.zeros(10)
        mfccs = filtered["MFCCs"].to_numpy()
        for utterance in mfccs:
            classification = classify_utterance(utterance, gmms)
            classifications[classification] += 1
        total_utterances = len(mfccs)
        classifications = classifications / total_utterances
        return np.round(classifications, 3)

    def compute_confusion_matrix(self, tied, covariance_type, use_kmeans):
        confusion = np.zeros((10, 10))
        gmms = []
        for digit in range(10):
            gmms.append(self.estimate_gmm_params(digit))
        for digit in range(10):
            classification = self.classify_all_utterances(digit, gmms)
            confusion[digit, :] = classification
        fig, ax = plt.subplots()
        ax.axis('off')  # Hide axis for a cleaner table display
        ax.table(cellText=confusion, loc='center', cellLoc='center', colLabels=[f"GMM {i}" for i in range(10)],
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
