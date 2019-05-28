#!/usr/bin/python
import re
import csv
import sys
import time
import random
import tweepy
import signal
import pandas as pd

consumer_key = "<removed>"
consumer_secret = "<removed>"
access_token = "<removed>"
access_token_secret = "<removed>"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)
all_tweet_ids = []

emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols
        u"\U0001F680-\U0001F6FF"  # map emojis
        u"\U0001F1E0-\U0001F1FF"  # iOS flags
                           "]+", flags=re.UNICODE)

def signal_handler(sig, frame):
	sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# each line is 19 bytes (18 chars + /r) -- test data -- "795952533516808192", "795952534565298177"
with open("first-debate.txt", "r") as f:
	for line in f.read().splitlines():
		all_tweet_ids.append(line)

# this splits the tweet ID's into 100 tweet segements
tweet_ids_split = [all_tweet_ids[x:x+100] for x in range(0, len(all_tweet_ids), 100)]
# chosen_tweet = random.choice(tweet_ids)
# print(chosen_tweet)

# function below gets tweets from API in batches of 100
# and outputs them to a file
def get_tweets(tweet_ids):
	iter_max = 1000
	counter = 0
	inner_counter = 0
	texts = []
	# tweet variable is list of 100 ids
	for tweet in tweet_ids:
		if counter < iter_max:
			counter = counter + 1
			time.sleep(1)
			public_tweets = api.statuses_lookup(tweet, trim_user=True)
			for tweet in public_tweets:
				if not tweet.retweeted and "RT @" not in tweet.text:
					cleaned = tweet.text.strip().replace("\n", "").replace("\u2026", "")
					try:
						cleaned.encode('ascii','ignore').decode('unicode-escape')
					except:
						pass
					cleaned = emoji_pattern.sub('', cleaned.encode("utf-8"))

					texts.append(cleaned)
					print(str(inner_counter) + "," + cleaned)
					inner_counter = inner_counter + 1
		else:
			return texts, len(texts)

out, collected_size = get_tweets(tweet_ids_split)
print(out)
print("Collected {} tweets".format(collected_size))
df = pd.DataFrame(out)
df.to_csv("tweets_out - test.csv", encoding="utf-8")
print("Success!")
