import numpy as np
from scipy.stats import multivariate_normal


def likelihood(gmm, utterance):
    components = gmm["components"]
    inner_sums = []
    for component in components:
        pi = component["pi"]
        mvn = multivariate_normal(mean=component["mean"], cov=component["covariance"])
        inner_sums.append(pi * mvn.pdf(utterance))
    col_sums = np.sum(np.array(inner_sums), axis=0)
    return np.sum(np.log(col_sums))


