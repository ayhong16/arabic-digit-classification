from analyzer import Analyzer

if __name__ == '__main__':
    analyzer = Analyzer()
    digit = 4
    n_components = analyzer.phoneme_map[digit]
    gmm_params = analyzer.estimate_gmm_params(digit, n_components, True, "full", False)
    analyzer.plot_likelihood_pdfs(gmm_params)

