import numpy as np
from scipy.stats import multivariate_normal

from GaussianMixture import GaussianMixture


def em(n_components, data, cov_type):
    gmm = GaussianMixture(n_components=n_components, covariance_type=cov_type, random_state=5)
    gmm.fit(data)
    labels = gmm.predict(data)
    means = gmm.means_
    covars = gmm.covariances_
    weights = gmm.weights_
    if len(covars) == 1:
        covars = np.tile(covars, n_components)
    cluster_info = {}
    for cluster_label in range(n_components):
        cluster_data = data[labels == cluster_label]
        cluster_info[tuple(means[cluster_label])] = {
            "cov": covars[cluster_label],
            "data": cluster_data,
            "pi": weights[cluster_label]
        }
    return cluster_info


def plot_em_contours(mean, cov, color, ax, coords):
    maxx_diff = max([abs(x[0] - mean[0]) for x in coords])
    maxy_diff = max([abs(x[1] - mean[1]) for x in coords])
    x = np.linspace(mean[0] - maxx_diff, mean[0] + maxx_diff, 1000)
    y = np.linspace(mean[1] - maxy_diff, mean[1] + maxy_diff, 1000)
    X, Y = np.meshgrid(x, y)
    z = multivariate_normal(mean, cov)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    ax.contour(X, Y, z.pdf(pos), colors=color, alpha=.6)


def em_component_gmm_helper(cluster_info):
    components = []
    for key in cluster_info:
        components.append({
            "pi": cluster_info[key]["pi"],
            "mean": np.array(key),
            "covariance": cluster_info[key]["cov"]
        })
    return components
