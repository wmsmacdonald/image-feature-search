from sklearn import cluster
import numpy as np


def get_time(times, num_votes_by_key):
    kmeans = cluster.KMeans(1)
    dbscan = cluster.DBSCAN()

    weighted_times = np.array([t for t, v in zip(times, num_votes_by_key)
                               for _ in range(0, v)])

    return int(np.median(weighted_times.reshape(-1, 1)))




