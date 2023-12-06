from matplotlib import pyplot as plt
from analyzer import Analyzer

if __name__ == '__main__':
    analyzer = Analyzer()
    comps = [(1, 2), (2, 3), (1, 3)]
    for d in range(10):
        analyzer.plot_kmeans_gmm(d, comps, analyzer.phoneme_map[d]["num_phonemes"])
    plt.show()
