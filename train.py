import pandas as pd
import numpy as np
from sklearn import preprocessing, svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import random

from sklearn.neighbors import KNeighborsClassifier

def preprocess(tweet):
    # this function should take in the tweet and remove features from 
    # it like URL's @handles and hashtags
    # returns the formatted tweet for further preprocessing
    pass


stop_words = stopwords.words("english")
wn_lemmatizer = WordNetLemmatizer()
# col 1 - 0: negative, 2: neutral, 4: positive

df = pd.read_csv("sentiment140/training.1600000.processed.noemoticon.csv", encoding='latin-1')
df = df.sample(frac=0.04)

# PREPROCESSING PART


# ML PART
X = np.array(df.drop(['sentiment', 'unknown', 'datetime', 'query', 'user'], 1))
y = np.array(df['sentiment'])

words = []
flattened_words = []

print(df.head())
for row in X:
    stop_words_removed = [word.lower() for word in row[0].split() if word not in (stop_words)]
    lemmatized = [wn_lemmatizer.lemmatize(tweet) for tweet in stop_words_removed]
    words.append(lemmatized)

flattened_words.append([' '.join(tweet) for tweet in words])
# print(len(flattened_words[0]))

tfv=TfidfVectorizer(sublinear_tf=True, stop_words = "english") # we need to give proper stopwords list for better performance
features=tfv.fit_transform(flattened_words[0])
print(features)


# https://www.kaggle.com/paoloripamonti/twitter-sentiment-analysis

# print("features: " + str(features.shape))
# print(y.shape)s
print("------------")


X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2)

clf = LogisticRegression()

clf.fit(X_train, y_train)
train_confidence = clf.score(X_train, y_train)
confidence = clf.score(X_test, y_test)
print("CONFIDENCE: " + str(confidence) + "\n" + "TRAIN CONFIDENCE: " + str(train_confidence))

# ------ MANUAL TESTING ------------
#  OK - 75%..!

# ---- real test set ----
# TEST_DF = pd.read_csv("sentiment140\\testdata.manual.2009.06.14.csv", encoding='latin-1')
# TEST_SENTIMENT = TEST_DF['sentiment']
# TEST_DATA = TEST_DF['tweet']
# -----------------------

TEST_DATA = ["I love Sklearn!", "Sentiment analysis is hard"]
TEST_WORDS = []
FLATTENED_TEST_WORDS = []
for sentence in TEST_DATA:
    remove_stop_words = [word.lower() for word in sentence.split() if word not in (stop_words)]
    lemmatized = [wn_lemmatizer.lemmatize(tweet) for tweet in remove_stop_words]
    TEST_WORDS.append(lemmatized)

FLATTENED_TEST_WORDS.append([' '.join(tweet) for tweet in TEST_WORDS])

TEST_FEATURES = tfv.transform(FLATTENED_TEST_WORDS[0])
prediction = clf.predict(TEST_FEATURES)
print(prediction)

# ---- below for real test set ----
# correct = 0
# wrong = 0
# total = len(TEST_SENTIMENT)
# for i in range(len(prediction)):
#     if prediction[i] == TEST_SENTIMENT[i]:
#         correct = correct + 1
#     elif TEST_SENTIMENT[i] == 2:
#         pass
#     else:
#         wrong = wrong + 1

# print(correct, wrong, total)
