from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection


def kfold(x, y, max_n):

    best_accuracy = 0
    best_n = 0

    for n in range(1, max_n):
        print('Running knn kfold: ', n, ' / ', max_n, end='\r')
        knn = KNeighborsClassifier(n_neighbors=n, n_jobs=-1)
        kfold = model_selection.KFold(n_splits=10)
        accuracy = model_selection.cross_val_score(knn, x, y, cv=kfold)
        if best_accuracy < accuracy.mean():
            best_accuracy = accuracy.mean()
            best_n = n

    return best_accuracy, best_n


def train(x_train, y_train, n):
    # create k nearest neighbors instance (object) with number of neighbors = 5
    knn = KNeighborsClassifier(n_neighbors=n)
    # train knn object using x_train and y_train
    knn.fit(x_train, y_train)
    return knn


def predict(x_test, knn):
    # predict using x_val
    y_pred_knn = knn.predict(x_test)
    return y_pred_knn
