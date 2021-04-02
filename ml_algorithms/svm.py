import numpy as np
from sklearn.svm import SVC
from sklearn import model_selection


def kfold(x, y):
    best_accuracy = 0
    best_kernel = ''
    svm_kernel = ['linear', 'poly', 'rbf']
    for kernel in svm_kernel:
        print('Running svc kfold with ', kernel, end='\r')
        svc = SVC(kernel=kernel)
        kfold = model_selection.KFold(n_splits=10)
        accuracy = model_selection.cross_val_score(svc, x, y, cv=kfold)
        if best_accuracy < accuracy.mean():
            best_accuracy = accuracy.mean()
            best_kernel = kernel

    return best_accuracy, best_kernel


def train(x_train, y_train, kernel):

    svc = SVC(kernel=kernel)
    # train mlp object using x_train and y_train
    svc.fit(x_train, y_train)
    return svc


def predict(x_test, knn):
    # predict using x_val
    y_pred_svc = knn.predict(x_test)
    return y_pred_svc

