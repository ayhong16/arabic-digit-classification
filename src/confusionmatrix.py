from matplotlib import pyplot as plt
from analyzer import Analyzer
import time

if __name__ == '__main__':
    analyzer = Analyzer()
    start = time.time()
    analyzer.compute_confusion_matrix(False, "full", True)
    end = time.time()
    print("Time to get confusion matrix: " + str(end - start))
    plt.show()
