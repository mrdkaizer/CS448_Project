from sklearn.decomposition import TruncatedSVD
from scipy.sparse import random as sparse_random


def reduction(n, X):
    # results = []
    # for i in range(, n):
    svd = TruncatedSVD(n_components=n, random_state=0)
    svd.fit(X)

    # print(svd.explained_variance_ratio_)
    res=0
    for k in svd.explained_variance_ratio_:
        res+=k
    # results.append(res)

    print(res)

    return svd.transform(X)
