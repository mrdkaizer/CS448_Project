from sklearn.cluster import KMeans
import numpy as np


def train(X):
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    return kmeans.labels_, kmeans


def test(X, kmeans):
    return kmeans.predict(X)