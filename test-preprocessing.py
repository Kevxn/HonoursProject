import re
import numpy as np
import pandas as pd

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

print(preprocess_tweet("This :) sentence  needs #cleaned @test http://t.co/3x4mpl3"))