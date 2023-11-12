import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kmeans import k_means, plot_contours, spherical_covar, diagonal_covar, full_covar
from util import likelihood
from sklearn.neighbors import KernelDensity


class DataParser:

    def __init__(self):
        self.df = pd.DataFrame(columns=['Digit', 'Index', 'Gender', 'MFCCs'])

    def parse_txt(self, filename, num_speakers):
        block_count = 1
        digit = 0
        index = 1
        iterations_per_gender = int((num_speakers / 2) * 10)
        iterations_per_digit = int(num_speakers * 10)
        gender = 'M'
        file = open('./spoken+arabic+digit/' + filename, 'r')
        lines = file.readlines()
        first_line = True
        mfccs = []
        for line in lines:
            if first_line:
                first_line = False
                continue

            stripped = line.strip()
            if len(stripped) == 0:  # new line signifying end of block
                new_row = pd.DataFrame([{'Digit': digit, 'Index': index, 'Gender': gender, 'MFCCs': mfccs}])
                self.df = pd.concat([self.df, new_row], ignore_index=True).reset_index(drop=True)
                mfccs = []
                if block_count % iterations_per_gender == 0:
                    gender = 'F' if gender == 'M' else 'M'
                if block_count % iterations_per_digit == 0:
                    digit += 1
                if index == 10:
                    index = 1
                else:
                    index += 1
                block_count += 1
            else:
                nums = stripped.split(" ")
                mfccs.append([float(num) for num in nums])

    def plot_timeseries(self, token, index, num_mfccs):
        metadata = self.get_single_utterance_data(token, index)
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
        metadata = self.get_single_utterance_data(token, index)
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

    def plot_gmms(self, token, comparisons, n_clusters):
        data = self.get_all_utterances(token)
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
                plot_contours(coords, axes[i], c)
                color_ind += 1
            axes[i].set_title("MFCC " + str(first_comp) + "(y) vs. MFCC" + str(second_comp) + "(x)")
        fig.suptitle("K-Means GMMs for Various 2D Plots for Digit " + str(token))
        plt.tight_layout()
        # plt.show()

    def compute_gmm_params(self, token, n_clusters, tied, covariance_type):
        data = self.get_all_utterances(token)
        cluster_info = k_means(n_clusters, data)
        cov_tied = "tied " if tied else "distinct "
        ret = {
            "digit": token,
            "number of components": n_clusters,
            "covariance type": cov_tied + covariance_type,
            "components": []
        }
        for key in cluster_info:
            component = {
                "pi": len(cluster_info[key]) / len(data),
                "mean": np.array(key),
            }
            if tied:
                input_data = data
            else:
                input_data = cluster_info[key]

            if covariance_type == "diagonal":
                covar = diagonal_covar(input_data)
            elif covariance_type == "full":
                covar = full_covar(input_data)
            else:
                covar = spherical_covar(input_data)
            component["covariance"] = covar
            ret["components"].append(component)
        return ret

    def plot_pdfs(self, gmm):
        fig, axs = plt.subplots(2, 5, tight_layout=True, sharex=True, sharey=True)

        for digit, ax in enumerate(axs.flatten()):
            filtered = self.df[self.df['Digit'] == digit]
            likelihoods = np.array(filtered['MFCCs'].apply(lambda x: likelihood(gmm, x))).reshape((-1,1))
            kde = KernelDensity(bandwidth=40, kernel='gaussian')
            kde.fit(likelihoods)
            x_values = np.linspace(min(likelihoods), max(likelihoods), 1000).reshape(-1, 1)
            log_pdf = kde.score_samples(x_values)
            ax.plot(x_values, np.exp(log_pdf))
            ax.set_title(f"PDF for {digit}")
        plt.tight_layout()
        plt.suptitle(f"PDFs Using GMM for {gmm["digit"]}")
        plt.show()

    def get_all_utterances(self, token):
        filtered = self.df[self.df['Digit'] == token]
        data = []
        for index, row in filtered.iterrows():
            data.append(row['MFCCs'])
        return np.vstack(data)

    def get_single_utterance_data(self, token, index):
        filtered = self.df[self.df['Digit'] == token]
        filtered = filtered[filtered['Index'] == index]
        metadata = filtered.iloc[0]  # first token
        data = metadata['MFCCs']
        x = [i for i in range(len(data))]
        return x, data
