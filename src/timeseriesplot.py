from matplotlib import pyplot as plt
from analyzer import Analyzer

if __name__ == '__main__':
    analyzer = Analyzer()
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(16, 14))
    for d, ax in enumerate(axes.flatten()):
        analyzer.plot_timeseries(d, 0, 13, ax)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.06, top=0.97, wspace=0.35, hspace=0.25)
    plt.show()
