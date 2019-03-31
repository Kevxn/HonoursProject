import os
import re
import nltk
import pickle
import numpy as np 
import pandas as pd
from nltk import sent_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()
tweet_tokenizer = TweetTokenizer()

def sentiment_analyzer_scores(sentence):
    # print("SENTENCE: ", sentence)
    new_sentence = ""
    for word in sentence:
        new_sentence = new_sentence + word + " "
    new_sentence = new_sentence.strip()
    # print(new_sentence)
    return analyser.polarity_scores(new_sentence)

def preprocess_tweet(tweet):
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

if os.name == "nt":
    df = pd.read_csv("sentiment140\\testdata.manual.2009.06.14.csv")
else:
    # df = pd.read_csv("sentiment140/testdata.manual.2009.06.14.csv")
    df = pd.read_csv("sentiment140/testdata.manual.2009.06.14.csv")

tweets = df['tweet']
sentiment = df['sentiment']

lemmatizer = WordNetLemmatizer()
stopwords = stopwords.words('english')

cleaned = []

correct = 0
wrong = 0

for i in range(len(tweets)):
    tweet = preprocess_tweet(tweets[i])
    
    stop_words_removed = [word.lower() for word in tweet_tokenizer.tokenize(tweet) if word not in (stopwords)]
    
    lemmatized = [lemmatizer.lemmatize(tweet, pos='v') for tweet in stop_words_removed]
    
    cleaned.append(lemmatized)
    # print(tweets[i])
    score_dict = sentiment_analyzer_scores(cleaned[i])
    compound_score = score_dict['compound']
    if sentiment[i] == 2:
        pass
    elif (compound_score > 0 and sentiment[i] == 4) or (compound_score < 0 and sentiment[i] == 0):
        correct = correct + 1
    else:
        wrong = wrong + 1

print("CORRECT: ", correct)
print("WRONG: ", wrong)
# VADER gets 72.15%
        

# print(cleaned)
# tokenzied_sent = sent_tokenize(tweets[0])
# print(tokenzied_sent)
