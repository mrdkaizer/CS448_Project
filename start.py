import pandas as pd
from data_processing import tf_idf
from ml_algorithms import knn

train = pd.read_csv("dataset/Data_Train.csv", header=0, delimiter=",", quoting=2)
test = pd.read_csv("dataset/Data_Test.csv", header=0, delimiter=",", quoting=2)

# tf-idf
data_features, vocab = tf_idf.clear_documents(train["STORY"], test['STORY'])

# knn kfold
# best_accuracy, best_n = knn.kfold(train_data_features, train["SECTION"], 20)
# print('best accuracy: ', best_accuracy)
# print('best neighbors: ', best_n)

# best_n = 12
# train_data_features = [] * train.shape[0]
train_data_features = data_features[:train.shape[0]]
test_data_features = data_features[train.shape[0]:]

knn_fit = knn.train(train_data_features, train["SECTION"], 12)
y_predict = knn.predict(test_data_features, knn_fit)
print(y_predict)

import xlsxwriter

# Create a workbook and add a worksheet.
workbook = xlsxwriter.Workbook('output.xlsx')
worksheet = workbook.add_worksheet()


# Start from the first cell. Rows and columns are zero indexed.
row = 1
col = 0

# Iterate over the data and write it out row by row.
worksheet.write(0,0,'SECTION')
for y in y_predict:
    worksheet.write(row, col,     y)
    row += 1


workbook.close()