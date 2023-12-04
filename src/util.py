import numpy as np
from scipy.stats import multivariate_normal
import ast


def compute_likelihood(gmm, utterance, selected):
    components = gmm["components"]
    inner_sums = []
    utterance = utterance[:, selected]
    for component in components:
        pi = component["pi"]
        mvn = multivariate_normal(mean=component["mean"], cov=component["covariance"])
        inner_sums.append(pi * mvn.pdf(utterance))
    col_sums = np.sum(np.array(inner_sums), axis=0)
    return np.sum(np.log(col_sums))


def classify_utterance(utterance, gmms):
    return np.argmax([compute_likelihood(gmm, utterance) for gmm in gmms])


def get_comp_covariance(x, y, covar):
    cov = np.zeros((2, 2))
    cov[0][0] = covar[x][x]
    cov[0][1] = covar[x][y]
    cov[1][0] = covar[y][x]
    cov[1][1] = covar[y][y]
    return cov
