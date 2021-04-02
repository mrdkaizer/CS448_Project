from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection


def kfold(x, y):
    best_accuracy = 0
    best_criterion = ''
    best_splitter = ''
    dtc_criterion = ['gini', 'entropy']
    dtc_splitter = ['best', 'random']
    for splitter in dtc_splitter:
        for criterion in dtc_criterion:
            print('Running dtc kfold with criterion: ', criterion, ', and splitter: ', splitter, end='\r')
            dtc = DecisionTreeClassifier(splitter=splitter, criterion=criterion)
            kfold = model_selection.KFold(n_splits=10)
            accuracy = model_selection.cross_val_score(dtc, x, y, cv=kfold)
            if best_accuracy < accuracy.mean():
                best_accuracy = accuracy.mean()
                best_criterion = criterion
                best_splitter = splitter

    return best_accuracy, best_criterion, best_splitter


def train(x_train, y_train, splitter, criterion):

    dtc = DecisionTreeClassifier(splitter=splitter, criterion=criterion)
    # train mlp object using x_train and y_train
    dtc.fit(x_train, y_train)
    return dtc


def predict(x_test, dtc):
    # predict using x_val
    y_pred_svc = dtc.predict(x_test)
    return y_pred_svc