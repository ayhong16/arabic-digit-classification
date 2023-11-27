from matplotlib import pyplot as plt

from analyzer import Analyzer

if __name__ == '__main__':
    analyzer = Analyzer()
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
    digit = 4
    n_components = phoneme_map[digit]
    gmm_params = analyzer.compute_gmm_params(digit, n_components, True, "full", False)
    analyzer.plot_likelihood_pdfs(gmm_params)

