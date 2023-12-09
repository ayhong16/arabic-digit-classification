import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal

from GaussianMixture import GaussianMixture


def em(mfccs, n_components, data, cov_type):
    gmm = GaussianMixture(n_components=n_components, covariance_type=cov_type, random_state=5)
    gmm.fit(data)
    labels = gmm.predict(data)
    means = gmm.means_
    covars = fix_em_cov_output(mfccs, gmm.covariances_, cov_type, n_components)
    weights = gmm.weights_
    cluster_info = {}
    for cluster_label in range(n_components):
        cluster_data = data[labels == cluster_label]
        cluster_info[tuple(means[cluster_label])] = {
            "cov": covars[cluster_label],
            "data": cluster_data,
            "pi": weights[cluster_label]
        }
    return cluster_info


def plot_em_contours(mean, cov, coords, ax=None):
    maxx_diff = max([abs(x[0] - mean[0]) for x in coords])
    maxy_diff = max([abs(x[1] - mean[1]) for x in coords])
    x = np.linspace(mean[0] - maxx_diff, mean[0] + maxx_diff, 1000)
    y = np.linspace(mean[1] - maxy_diff, mean[1] + maxy_diff, 1000)
    X, Y = np.meshgrid(x, y)
    z = multivariate_normal(mean, cov)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    if ax is None:
        plt.contour(X, Y, z.pdf(pos), colors='black', alpha=.5)
    else:
        ax.contour(X, Y, z.pdf(pos), colors='black', alpha=.5)


def em_component_gmm_helper(cluster_info):
    components = []
    for key in cluster_info:
        components.append({
            "pi": cluster_info[key]["pi"],
            "mean": np.array(key),
            "covariance": cluster_info[key]["cov"]
        })
    return components


def fix_em_cov_input(tied, cov_type):
    if tied:
        if cov_type == "full":
            return "tied"
        if cov_type == "diag":
            return "tied_diag"
        if cov_type == "spherical":
            return "tied_spherical"
    else:
        return cov_type


def fix_em_cov_output(mfccs, covariance, covariance_type, n_components):
    mfcc_length = len(mfccs)
    if covariance_type == "full":
        ret = covariance
    elif covariance_type == "tied":
        ret = np.asarray([covariance] * n_components)
    elif covariance_type == "tied_diag" or covariance_type == "diag":
        ret = np.asarray([np.diag(cov) for cov in covariance])
    elif covariance_type == "tied_spherical":
        ret = np.asarray([np.diag([cov] * mfcc_length) for cov in covariance])
    elif covariance_type == "spherical":
        ret = np.asarray([np.diag([cov] * mfcc_length) for cov in covariance])
    else:
        ret = ValueError("Invalid covariance type: " + covariance_type)
    assert len(ret.shape) == 3
    assert ret.shape[0] == n_components
    assert ret.shape[1] == mfcc_length
    assert ret.shape[2] == mfcc_length
    return ret
