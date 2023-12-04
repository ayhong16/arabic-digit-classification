from analyzer import Analyzer

if __name__ == '__main__':
    analyzer = Analyzer()
    digit = 4
    gmm_params = analyzer.estimate_gmm_params(digit, [0, 7, 9, 12])
    analyzer.plot_likelihood_pdfs(gmm_params)

