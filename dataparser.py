import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kmeans import k_means, plot_contours


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
        mapped_data = {}
        for i in range(13):
            mapped_data[i + 1] = []
        for x in data:
            for i in range(13):
                mapped_data[i + 1].append(x[i])
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


if __name__ == '__main__':
    parser = DataParser()
    parser.parse_txt("Train_Arabic_Digit.txt", 66)
    comps = [(1, 2), (2, 3), (1, 3)]
    phoneme_map = {
        0: 3,
        1: 3,
        2: 3,
        3: 4,
        4: 5,
        5: 4,
        6: 3,
        7: 3,
        8: 3,
        9: 4,
    }
    for d in range(10):
        parser.plot_gmms(d, comps, phoneme_map[d])
        # plt.figure(d)
        # parser.plot_timeseries(digit, 1, 3)
        # parser.plot_scatter(d, 1, comp)
    plt.show()
