import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from statistics import mean, stdev
from data_processing import tf_idf, svd


train = pd.read_csv("dataset/Data_Train.csv", header=0, delimiter=",", quoting=2)

# tf-idf
data_features, vocab = tf_idf.clear_documents(train["STORY"], None)


# split data correctly from tf-idf
X = svd.reduction(20, data_features)

accuracy = [[], [], [], []]
names = ["KNN", "SVM", "DTC", "MLP"]
for run in range(10):
    print(run)
    # split data to train (60%) test (20%), validation (20%)
    x_train, x_test, y_train, y_test = train_test_split(X, train["STORY"], test_size=0.4)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)

    # create k nearest neighbors instance (object) with number of neighbors = 5
    print('running knn')
    knn = KNeighborsClassifier(n_neighbors=6)
    # train knn object using x_train and y_train
    knn.fit(x_train, y_train)
    # predict using x_val
    y_pred_knn = knn.predict(x_val)
    accuracy[0].append(accuracy_score(y_val, y_pred_knn))
    print('running svc')
    # create support vector classifier (object) without parameters
    svc = SVC(kernel='rbf')
    # train svc object using x_train and y_train
    svc.fit(x_train, y_train)
    # predict using x_val
    y_pred_svm = svc.predict(x_val)
    accuracy[1].append(accuracy_score(y_val, y_pred_svm))

    print('running dtc')
    # create decision tree classifier (object) without parameters
    dtc = DecisionTreeClassifier(splitter='best', criterion='entropy')
    # train dtc object using x_train and y_train
    dtc.fit(x_train, y_train)
    # predict using x_val
    y_pred_dtc = dtc.predict(x_val)
    accuracy[2].append(accuracy_score(y_val, y_pred_dtc))
    print('running mlp')
    # create random forest classifier (object) without parameters
    mlp = MLPClassifier(max_iter=2000,
                        hidden_layer_sizes=80)
    # train rfc object using x_train and y_train
    mlp.fit(x_train, y_train)
    # predict using x_val
    y_pred_rfc = mlp.predict(x_val)
    accuracy[3].append(accuracy_score(y_val, y_pred_rfc))



knn_mean = mean(accuracy[0])
knn_stdev = stdev(accuracy[0])
svm_mean = mean(accuracy[1])
svm_stdev = stdev(accuracy[1])
dtc_mean = mean(accuracy[2])
dtc_stdev = stdev(accuracy[2])
mlp_mean = mean(accuracy[3])
mlp_stdev = stdev(accuracy[3])
print("[KNN] Mean: " + str(knn_mean) + ", Stdev: " + str(knn_stdev))
print("[SVM] Mean: " + str(svm_mean) + ", Stdev: " + str(svm_stdev))
print("[DTC] Mean: " + str(dtc_mean) + ", Stdev: " + str(dtc_stdev))
print("[MLP] Mean: " + str(mlp_mean) + ", Stdev: " + str(mlp_stdev))
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(accuracy)
ax.set_xticklabels(names)
plt.show()