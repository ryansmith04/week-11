"""K-means clustering on the seaborn diamonds dataset."""


from time import time

import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans

# load the data and keep only numerical columns
diamonds = sns.load_dataset("diamonds")
diamonds_numeric = diamonds.select_dtypes(include="number")


def kmeans(X, k):
    """
    Run k-means clustering on a numerical Numpy array 'X'.

    Returns a tuple (centroids, labels):
    - centroids has shape (k, n_features), one row per cluster center.
    - labels has shape (n_samples,), the cluster for each row of X.
    """
    # Create a KMeans instance with k clusters
    model = KMeans(n_clusters=k, random_state=0)

    # Fit the model to the data
    model.fit(X)

    return model.cluster_centers_, model.labels_


def kmeans_diamonds(n, k):
    """Run kmeans() with `k` clusters on the first `n` diamonds rows."""
    X = diamonds_numeric.iloc[:n].to_numpy()
    return kmeans(X, k)


def kmeans_timer(n, k, n_iter=5):
    """
    Run kmeans_diamonds(n, k) 'n_iter' times.

    Return the average time taken in seconds.
    """
    times = []
    for _ in range(n_iter):
        start = time()
        kmeans_diamonds(n, k)
        times.append(time() - start)
    return np.mean(times)
