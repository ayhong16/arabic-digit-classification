from analyzer import Analyzer

if __name__ == '__main__':
    analyzer = Analyzer()
    digit = 4
    gmm_params = analyzer.estimate_gmm_params(digit)
    analyzer.plot_likelihood_pdfs(gmm_params)

