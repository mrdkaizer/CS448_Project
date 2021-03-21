import pandas as pd
from data_processing import tf_idf
from ml_algorithms import knn

train = pd.read_csv("dataset/Data_Train.csv", header=0, delimiter=",", quoting=2)
all_document = ''

train_data_features, tf_idf_vocab = tf_idf.clear_documents(train)
print(tf_idf_vocab)

best_accuracy, best_n = knn.kfold(train_data_features, train["SECTION"], 5)
print('best accuracy: ', best_accuracy)
print('best neighbors: ', best_n)

