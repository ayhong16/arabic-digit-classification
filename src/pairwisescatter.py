from matplotlib import pyplot as plt
from analyzer import Analyzer

if __name__ == '__main__':
    analyzer = Analyzer()
    comps = [(1, 2), (2, 3), (1, 3)]
    for d in range(10):
        plt.figure(d)
        analyzer.plot_scatter(d, 1, comps)
    plt.show()
