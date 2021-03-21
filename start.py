import pandas as pd
from data_processing import tf_idf


train = pd.read_csv("dataset/Data_Train.csv", header=0, delimiter=",", quoting=2)
all_document = ''

train_data_features, tfidf_vocab = tf_idf.clear_documents(train)
print(tfidf_vocab)
