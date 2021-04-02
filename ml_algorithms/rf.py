from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection


def kfold(x, y):

    best_accuracy = 0

    rf_criterion = ['gini', 'entropy']

    for criterion in rf_criterion:
        for max_depth in range(50, 500, 50):
            print('Running random forest kfold with criterion: ', criterion, ' and max depth: ', max_depth, end='\r')
            knn = RandomForestClassifier(criterion=criterion, max_depth=max_depth)
            kfold = model_selection.KFold(n_splits=10)
            accuracy = model_selection.cross_val_score(knn, x, y, cv=kfold)
            if best_accuracy < accuracy.mean():
                best_accuracy = accuracy.mean()
                best_criterion = criterion
                best_depth = max_depth

    return best_accuracy, best_criterion, best_depth


def train(X, Y, depth, criterion):
    rf = RandomForestClassifier(criterion=criterion, max_depth=depth).fit(X, Y)
    return rf


def test(X, rf):
    return rf.predict(X)