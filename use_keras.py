import pandas as pd
import numpy as np
from sklearn import preprocessing, svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
import random
import gensim
import re
import os
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Flatten
from tensorflow.keras.models import Sequential, save_model, load_model

model_name = "full_8epoch_64_batch.h5"

if os.name == "nt":
    with open("F:\\Hons Project\\models\\TfidfVectorizer" + model_name + ".pickle", "rb") as f:
        tfv = pickle.load(f)
    model = load_model("F:\\Hons Project\\models\\" + model_name)
else:
    with open("/models/TfidfVectorizer" + model_name + ".pickle", "rb") as f:
        tfv = pickle.load(f)
    model = load_model("/models/" + model_name)

stop_words = stopwords.words('english')
wn_lemmatizer = WordNetLemmatizer()

def preprocess_tweet(tweet):
    # remove emoticons
    tweet = re.sub('[:]([\(\)\/\\\[\]]|[A-Za-z1-9@])', '', tweet)
    # remove urls
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', tweet)
    # remove usernames
    tweet = re.sub('@[^\s]+', '', tweet)
    # remove whitespaces
    tweet = re.sub('[\s]+', ' ', tweet)
    # remove '#'s
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    return tweet

def get_sentiment(sentiment_vector):
    neg, pos = list(sentiment_vector)
    if pos > neg:
        return "Positive ({}%)".format(pos * 100)
        # return 4
    else:
        # return 0
        return "Negative ({}%)".format(neg * 100)

words = []
flattened_words = []

""" for manual testing
df = pd.read_csv("C:\\Users\\kevin\\Documents\\Hons Project\\scripts\\sentiment140\\testdata.manual.2009.06.14.csv")
sentiment = df['sentiment']
tweet_text = df['tweet']
"""

TEST_DATA = ["Playing football is fun", "Android Studio is the worst"]

for row in TEST_DATA:
    tweet = preprocess_tweet(row)
    stop_words_removed = [word.lower() for word in tweet.split() if word not in (stop_words)]
    lemmatized = [wn_lemmatizer.lemmatize(tweet) for tweet in stop_words_removed]
    words.append(lemmatized)

flattened_words.append([' '.join(tweet) for tweet in words])

features=tfv.transform(flattened_words[0])

predictions = []
prediction = model.predict([features])
for pred in prediction:
    prd = get_sentiment(pred)
    predictions.append(prd)
    print(prd)

""" for manual testing
correct = 0
wrong = 0
total = len(sentiment)

for i in range(len(prediction)):
    if predictions[i] == sentiment[i]:
        correct = correct + 1
    elif sentiment[i] == 2:
        pass
    else:
        wrong = wrong + 1

print(correct, wrong, total)
"""