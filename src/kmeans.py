import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans


def k_means(n_clusters, data):
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=np.random.RandomState(42), init="k-means++")
    kmeans.fit(data)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    cluster_info = {}
    for cluster_label in range(n_clusters):
        cluster_data = data[labels == cluster_label]
        cluster_info[tuple(centers[cluster_label])] = cluster_data
    return cluster_info


def spherical_covar(data):
    num_dims = data.shape[1]
    col_stack = data.reshape((len(data) * 13, 1))
    cov = np.cov(col_stack, rowvar=False)
    return np.identity(num_dims) * cov


def diagonal_covar(data):
    covars = []
    num_dims = data.shape[1]
    for dim in range(13):
        cov = np.cov([d[dim] for d in data])
        covars.append(cov)
    return np.identity(num_dims) * np.array(covars)


def full_covar(data):
    return np.cov(data, rowvar=False)


def plot_kmeans_contours(data, ax, color):
    mean = np.mean(data, axis=0)
    maxx_diff = max([abs(x[0] - mean[0]) for x in data])
    maxy_diff = max([abs(x[1] - mean[1]) for x in data])
    x = np.linspace(mean[0] - maxx_diff, mean[0] + maxx_diff, 1000)
    y = np.linspace(mean[1] - maxy_diff, mean[1] + maxy_diff, 1000)
    X, Y = np.meshgrid(x, y)
    cov = np.cov(data, rowvar=False)
    z = multivariate_normal(mean, cov)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    ax.contour(X, Y, z.pdf(pos), colors=color, alpha=.6)