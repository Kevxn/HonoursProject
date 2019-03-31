import re
import time
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
import pickle

from sklearn.neighbors import KNeighborsClassifier

time1 = time.time()

def preprocess(tweet):
    # this function should take in the tweet and remove features from 
    # it like URL's @handles and hashtags
    # returns the formatted tweet for further preprocessing
    
    # remove emoticons
    tweet = re.sub('[:]([\(\)\/\\\[\]]|[A-Za-z1-9@])', '', tweet)
    # remove urls
    tweet = re.sub('(www|http)\S+', '', tweet)
    # remove usernames
    tweet = re.sub('@[^\s]+', '', tweet)
    # remove whitespaces
    tweet = re.sub('[\s]+', ' ', tweet)
    # remove '#'s
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    return tweet


stop_words = stopwords.words("english")
wn_lemmatizer = WordNetLemmatizer()
# col 1 - 0: negative, 2: neutral, 4: positive

df = pd.read_csv("sentiment140/training.1600000.processed.noemoticon.csv", encoding='latin-1')
df = df.sample(frac=0.99)

# PREPROCESSING PART


# ML PART
X = np.array(df.drop(['sentiment', 'unknown', 'datetime', 'query', 'user'], 1))
y = np.array(df['sentiment'])

words = []
flattened_words = []

print(df.head())
for row in X:
    tweet = preprocess(row[0])
    stop_words_removed = [word.lower() for word in tweet.split() if word not in (stop_words)]
    lemmatized = [wn_lemmatizer.lemmatize(tweet, pos='v') for tweet in stop_words_removed]
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
print(features.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2)

# clf = LogisticRegression()
clf = svm.LinearSVC(C=0.1)

clf.fit(X_train, y_train)
train_confidence = clf.score(X_train, y_train)
confidence = clf.score(X_test, y_test)
print("CONFIDENCE: " + str(confidence) + "\n" + "TRAIN CONFIDENCE: " + str(train_confidence))

# ------ MANUAL TESTING ------------
#  OK - 75%..!

# ---- real test set ----
TEST_DF = pd.read_csv("sentiment140\\testdata.manual.2009.06.14.csv", encoding='latin-1')
TEST_SENTIMENT = TEST_DF['sentiment']
TEST_DATA = TEST_DF['tweet']
# -----------------------

# TEST_DATA = ["I love Sklearn!", "Sentiment analysis is hard"]
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
correct = 0
wrong = 0
total = len(TEST_SENTIMENT)
for i in range(len(prediction)):
    if prediction[i] == TEST_SENTIMENT[i]:
        correct = correct + 1
    elif TEST_SENTIMENT[i] == 2:
        pass
    else:
        wrong = wrong + 1

print(correct, wrong, total)
print("Script ran in {} seconds".format(time.time() - time1))
# 79.78% logistic regression test accuracy (99% of training dataset) average of 5 runs
# 0.7577% accuracy LinearSVC test accuracy (99% of training dataset)
# 80.78% accuracy LinearSVC C=0.1 test accuracy (10% of training dataset) (79.44% best of 5)

model_name = "SVM0.99datasetC0.1.h5"
vectorizer_filename = "F:\\Hons Project\\models\\TfidfVectorizer" + model_name + ".pickle"

with open(vectorizer_filename, "wb") as f:
    pickle.dump(tfv, f)

with open("F:\\Hons Project\\models\\" + model_name, "wb") as f:
    pickle.dump(clf, f)
