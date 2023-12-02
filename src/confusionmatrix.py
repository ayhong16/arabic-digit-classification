import time

from analyzer import Analyzer

if __name__ == '__main__':
    analyzer = Analyzer()
    analyzer.compute_confusion_matrix(False, "full", False)
