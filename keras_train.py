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
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Flatten
from tensorflow.keras.models import Sequential, save_model



def preprocess_tweet(tweet):
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'url', tweet)
    #convert all @username to "AT_USER"
    tweet = re.sub('@[^\s]+', 'user', tweet)
    #fix white spacing
    tweet = re.sub('[\s]+', ' ', tweet)
    #strip '#' from hashtags
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    return tweet

stop_words = stopwords.words("english")
wn_lemmatizer = WordNetLemmatizer()
# col 1 - 0: negative, 2: neutral, 4: positive
df = pd.read_csv("sentiment140/training.1600000.processed.noemoticon.csv", encoding='latin-1')
df = df.sample(frac=0.5)

X = np.array(df.drop(['sentiment', 'unknown', 'datetime', 'query', 'user'], 1))
y = np.array(df['sentiment'])
words = []
flattened_words = []

for row in X:
    tweet = preprocess_tweet(row[0])
    stop_words_removed = [word.lower() for word in tweet.split() if word not in (stop_words)]
    lemmatized = [wn_lemmatizer.lemmatize(tweet) for tweet in stop_words_removed]
    words.append(lemmatized)
    
max_words = 5000
flattened_words.append([' '.join(tweet) for tweet in words])

tfv=TfidfVectorizer(sublinear_tf=True, stop_words = "english") # we need to give proper stopwords list for better performance
features=tfv.fit_transform(flattened_words[0])

y = pd.get_dummies(df['sentiment']).values
embed_dim = 128
lstm_out = 196

model = Sequential()
# model.add(Embedding(max_words, embed_dim, input_length = X.shape[1]))
model.add(Dense(128, activation='relu', input_dim=features.shape[1]))
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
print(model.summary())

print(features.shape, y.shape)
print("X: {}".format(features))
print("y: {}".format(y))

X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.3, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
batch_size = 32
model.fit(X_train, y_train, epochs = 4, batch_size=batch_size)

score, acc = model.evaluate(X_test, y_test, verbose=2, batch_size=batch_size)
print("score: {}".format(score))
print("acc: {}".format(acc))

save_model(model, "C:\\Users\Kevin\\Desktop\\modelstemp\\modelHalfDataset.h5", True, True)
