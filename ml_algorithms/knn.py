from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def kfold(x, y, max_n):
    # X_train, X_test, y_train, y_test = train_test_split(tfidf_train_data_features, train["SECTION"], test_size=0.2,
    # random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    best_accuracy = 0
    best_n = 0

    for n in range(1, max_n):
        knn = KNeighborsClassifier(n_neighbors=n, n_jobs=-1)
        kfold = model_selection.KFold(n_splits=10)
        accuracy = model_selection.cross_val_score(knn, X_train, y_train, cv=kfold)
        if best_accuracy < accuracy.mean():
            best_accuracy = accuracy.mean()
            best_n = n

    return best_accuracy, best_n

