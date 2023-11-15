from matplotlib import pyplot as plt
from analyzer import Analyzer

if __name__ == '__main__':
    analyzer = Analyzer()
    for d in range(10):
        plt.figure(d)
        analyzer.plot_timeseries(d, 1, 3)
    plt.show()
