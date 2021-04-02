# dimensionality reduction KNN
# plot_accuracy = []
# for f in range(2, 100):
#     reduced_features = svd.reduction(f, data_features)
#
#     train_data_features = reduced_features[:train.shape[0]]
#     test_data_features = reduced_features[train.shape[0]:]
#
#     # knn kfold
#     best_accuracy, best_n = knn.kfold(train_data_features, train["SECTION"], 20)
#     print('best accuracy: ', best_accuracy)
#     print('best neighbors: ', best_n)
#     plot_accuracy.append(best_accuracy)
#
# show_plot(plot_accuracy, range(2, 100), 'number of features', 'training accuracy')


# dimensionality reduction MLP
# plot_accuracy = []
# for f in range(2, 100):
#     reduced_features = svd.reduction(f, data_features)
#
#     train_data_features = reduced_features[:train.shape[0]]
#     test_data_features = reduced_features[train.shape[0]:]
#
#     # knn kfold
#     best_accuracy, best_n = mlp.kfold(train_data_features, train["SECTION"], 600)
#     print('best accuracy: ', best_accuracy)
#     print('best neighbors: ', best_n)
#     plot_accuracy.append(best_accuracy)
#
# show_plot(plot_accuracy, range(2, 100), 'number of features', 'training accuracy')
#
