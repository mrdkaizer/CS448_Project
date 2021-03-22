from sklearn.neural_network import MLPClassifier
from sklearn import model_selection
from sklearn.model_selection import train_test_split


def kfold(x, y, max_iters):
    print('test from kfold ')
    best_accuracy = 0
    best_hidden_layer_size = 0

    for hidden_layer_size in range(200, 500, 50):
        print('Running mlp kfold: ', hidden_layer_size, ' / ', 500)
        mlp = MLPClassifier(random_state=1, max_iter=max_iters, hidden_layer_sizes=hidden_layer_size)
        kfold = model_selection.KFold(n_splits=10)
        accuracy = model_selection.cross_val_score(mlp, x, y, cv=kfold)
        if best_accuracy < accuracy.mean():
            best_accuracy = accuracy.mean()
            best_hidden_layer_size = hidden_layer_size

    return best_accuracy, best_hidden_layer_size


def train(x_train, y_train, max_iters, hidden_layer_size):

    mlp = MLPClassifier(random_state=1,
                        max_iter=max_iters,
                        hidden_layer_sizes=hidden_layer_size
                        )
    # train mlp object using x_train and y_train
    mlp.fit(x_train, y_train)
    return mlp


def predict(x_test, knn):
    # predict using x_val
    y_pred_knn = knn.predict(x_test)
    return y_pred_knn