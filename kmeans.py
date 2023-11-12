import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans


def k_means(n_clusters, data):
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto')
    kmeans.fit(data)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    cluster_info = {}
    for point, label in zip(data, labels):
        mean = tuple(centers[label])
        if mean not in cluster_info:
            cluster_info[mean] = []
        cluster_info[mean].append(point)
    return cluster_info


def plot_contours(data, ax, color):
    mean = np.mean(data, axis=0)
    x = np.linspace(mean[0] - 5, mean[0] + 5, 1000)
    y = np.linspace(mean[1] - 5, mean[1] + 5, 1000)
    X, Y = np.meshgrid(x, y)
    cov = np.cov(data, rowvar=False)
    z = multivariate_normal(mean, cov)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    ax.contour(X, Y, z.pdf(pos), colors=color, alpha=.6)


def spherical_covar(data):
    cov = np.cov(data, rowvar=False)
    return np.identity(2) * cov


def diagonal_covar(data):
    covars = []
    for dim in range(13):
        cov = np.cov([d[dim] for d in data])
        covars.append(cov)
    return np.identity(13) * np.array(covars)


def full_covar(data):
    return np.cov(data, rowvar=False)
