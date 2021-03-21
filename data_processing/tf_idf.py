import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


stemmer = SnowballStemmer('english')
stops = set(stopwords.words("english"))


def clear_text(input_text):
    # 1. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", input_text)

    # 2. Convert to lower case, split into individual words
    words = letters_only.lower().split()

    # 3. Remove stop words
    meaningful_words = [w for w in words if not w in stops]

    # 4. Stem words
    stemmed_meaningful_words = [stemmer.stem(w) for w in
                                meaningful_words]

    # 5. Join the words back into one string separated by space,
    # and return the result.
    return " ".join(stemmed_meaningful_words)


def clear_documents(train):
    # Get the number of reviews based on the dataframe column size
    length = train["STORY"].size

    # Initialize an empty list to hold the clean reviews
    clean_train = []

    # Loop over each review; create an index i that goes from 0 to
    # the length of the movie review list
    for i in range(length):
        # Call our function for each one, and add the result to the
        # list of clean reviews
        clean_train.append(clear_text(train["STORY"][i]))

    # Initialize the "TfidfVectorizer" object, which is scikitlearn's tf/idf tool.
    tfidf_vectorizer = TfidfVectorizer(max_df=1.0, \
                                       max_features=25000, \
                                       min_df=0.01, \
                                       stop_words=None, \
                                       use_idf=True, \
                                       tokenizer=None, \
                                       ngram_range=(1, 3))
    # Tf-idf-weighted term-document sparse matrix
    tfidf_train_data_features = tfidf_vectorizer.fit_transform(clean_train)

    # Convert the result to nampy array
    tfidf_train_data_features = tfidf_train_data_features.toarray()
    #print('Features = ', tfidf_train_data_features.shape)  # (25000, 48)

    # Take a look at the words in the vocabulary
    tfidf_vocab = tfidf_vectorizer.get_feature_names()

    return tfidf_train_data_features, tfidf_vocab
