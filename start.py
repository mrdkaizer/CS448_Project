import pandas as pd
from data_processing import tf_idf, svd
from ml_algorithms import knn, mlp, svm, dtc, rf
from plot import show_plot

train = pd.read_csv("dataset/Data_Train.csv", header=0, delimiter=",", quoting=2)
test = pd.read_csv("dataset/Data_Test.csv", header=0, delimiter=",", quoting=2)

# tf-idf
data_features, vocab = tf_idf.clear_documents(train["STORY"], test['STORY'])

# split data correctly from tf-idf
reduced_features = svd.reduction(20, data_features)
train_data_features = reduced_features[:train.shape[0]]
test_data_features = reduced_features[train.shape[0]:]


# knn kfold
# best_accuracy, best_n = knn.kfold(train_data_features, train["SECTION"], 20)
# print('best accuracy: ', best_accuracy)
# print('best neighbors: ', best_n)


best_accuracy, best_hidden_layer_size = mlp.kfold(train_data_features, train["SECTION"], 2000)
print('best accurancy:', best_accuracy)
print('best hidden lays ', best_hidden_layer_size)
# best_hidden_layer_size=300
# mlp_fit = mlp.train(train_data_features, train["SECTION"], 600, best_hidden_layer_size)
# y_predict = knn.predict(test_data_features, mlp_fit)



best_accuracy, best_kernel = svm.kfold(train_data_features, train["SECTION"])
print('best accurancy:', best_accuracy)
print('best kernel:', best_kernel)
# svm_fit = svm.train(train_data_features, train["SECTION"], best_kernel)
# y_predict = svm.predict(test_data_features, svm_fit)


best_accuracy, best_criterion, best_splitter = dtc.kfold(train_data_features, train["SECTION"])
print('best accurancy: ', best_accuracy)
print('best criterion: ', best_criterion)
print('best splitter', best_splitter)
# dtc_fit = dtc.train(train_data_features, train["SECTION"], best_splitter, best_criterion)
# y_predict = svm.predict(test_data_features, dtc_fit)



best_accuracy, best_criterion, best_depth = rf.kfold(train_data_features, train["SECTION"])
print('best accurancy: ', best_accuracy)
print('best criterion: ', best_criterion)
print('best depth', best_depth)
# rf_fit = rf.train(train_data_features, train["SECTION"], best_depth, best_criterion)
# y_predict = svm.predict(test_data_features, rf_fit)










# import xlsxwriter

# # Create a workbook and add a worksheet.
# workbook = xlsxwriter.Workbook('output.xlsx')
# worksheet = workbook.add_worksheet()


# # Start from the first cell. Rows and columns are zero indexed.
# row = 1
# col = 0

# # Iterate over the data and write it out row by row.
# worksheet.write(0,0,'SECTION')
# for y in y_predict:
#     worksheet.write(row, col, y)
#     row += 1


# workbook.close()