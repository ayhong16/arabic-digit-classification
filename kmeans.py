import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans


def k_means(n_clusters, data):
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto')
    kmeans.fit(data)
    labels = kmeans.labels_
    cluster_info = {}
    for cluster_label in range(n_clusters):
        cluster_data = data[labels == cluster_label]
        cluster_info[cluster_label] = cluster_data
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
