from matplotlib import pyplot as plt
from analyzer import Analyzer

if __name__ == '__main__':
    analyzer = Analyzer()
    comps = [(1, 2), (2, 3), (1, 3)]
    phoneme_map = {
        0: 4,
        1: 4,
        2: 3,
        3: 4,
        4: 3,
        5: 3,
        6: 4,
        7: 3,
        8: 4,
        9: 3,
    }
    for d in range(10):
        analyzer.plot_kmeans_gmm(d, comps, phoneme_map[d])
    plt.show()
