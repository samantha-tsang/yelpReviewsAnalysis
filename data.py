############ Code for data collection and preprocessing ############

########## Source ##########
# https://towardsdatascience.com/load-yelp-reviews-or-other-huge-json-files-with-ease-ad804c2f1537

import numpy as np
import pandas as pd
import re

import gensim
from sklearn.model_selection import train_test_split
from gensim.models.phrases import Phrases, Phraser
from nltk import bigrams
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from gensim.models import ldamulticore

from itertools import chain
from collections import Counter
import json

##### Load data
# load yelp reviews and business data sets (from yelp.com)
business_df = pd.read_json("yelp_academic_dataset_business.json", lines = True)
business_df.to_csv("business_df.csv")

review_df = pd.read_json("yelp_academic_dataset_review.json", lines=True,
                         dtype = {'review_id':str,'user_id':str,
                         'business_id':str,'stars':int,
                         'date':str,'text':str,'useful':int,
                         'funny':int,'cool':int},
                         chunksize = 1000000)

##### clean business df
business_df = business_df[business_df["is_open"] == 1] # only use open business
columns_to_drop = ["postal_code", "latitude", "longitude", "review_count", "hours"] # drop unnecessary columns
business_df = business_df.drop(columns_to_drop, axis = 1)
restaurants_df = business_df[business_df['categories'].str.contains('Restaurants|Food',
                                                                    case=False, na=False)] # only use restaurants
##### read in and merge in each chunks from reviews reader to business df
chunk_list = []
for chunk_review in review_df:
    chunk_review = chunk_review.drop(['review_id','useful','funny','cool'], axis=1) # drop unnecessary columns
    chunk_review = chunk_review.rename(columns={'stars': 'review_stars'}) # rename column to avoid confusion with other
    chunk_merged = pd.merge(restaurants_df, chunk_review, on='business_id', how='inner') # inner join with business to match business id
    # Show feedback on progress
    print(f"{chunk_merged.shape[0]} out of {1000000:,} related reviews")
    chunk_list.append(chunk_merged)
restaurant_reviews = pd.concat(chunk_list, ignore_index=True, join='outer', axis=0)
restaurant_reviews[restaurant_reviews["text"].isna()]
restaurant_reviews = restaurant_reviews.drop([1118960, 1118961], axis = 0) # drop rows without reviews
restaurant_reviews = restaurant_reviews.reset_index(drop = True)
restaurant_reviews.to_csv("restaurant_reviews.csv")

##### preprocess data
restaurant_reviews = restaurant_reviews[(restaurant_reviews["state"] == "CA") |
                                        (restaurant_reviews["state"] == "FL")] # only use reviews from restaurants in california and florida, ~ 800,000
reviews = restaurant_reviews["text"]

# get rid of symbols
def remove_symbols(string):
    string = re.sub("\n", " ", string) # replace \n symbol with space
    new_string = re.sub("[^a-zA-Z0-9 ]", "", string) # take away all symbols
    return new_string
remove_symbols(reviews[0]) # try function
reviews = reviews.map(remove_symbols)
restaurant_reviews = pd.concat([restaurant_reviews, reviews], axis = 1)
restaurant_reviews.columns = ['Unnamed: 0.1', 'Unnamed: 0', 'business_id', 'name', 'address', 'city',
       'state', 'stars', 'is_open', 'attributes', 'categories', 'user_id',
       'review_stars', 'text', 'date', 'text_no_symbol']
restaurant_reviews.to_csv("restaurant_reviews.csv")

# function for stemming and tokenize words
stemmer = SnowballStemmer(language='english')
def lemmatize_stemming(token):
    # input: token word
    # output stemmed/lemmatized token
    return stemmer.stem(WordNetLemmatizer().lemmatize(token, pos='v'))

# function to remove stopwords
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    # input string
    # output: list of tokenized words in string with stop words removed
    filtered_sentence = [w for w in gensim.utils.simple_preprocess(text) if w.lower() not in stop_words]
    return filtered_sentence

# combine two functions for preprocessing
def preprocess(text):
    # input string
    # output: list of tokenized words, stemmed and lemmatized
    result = [lemmatize_stemming(token) for token in remove_stopwords(text)]
    return result
preprocess(reviews[65]) # test function
reviews[0:2].map(preprocess)

# test function on time
import time
start = time.time()
reviews[0:10000].map(preprocess)
end = time.time()
print("Run time on 10000 samples: " + str(end - start) + "secs") # about 15 seconds on 10k samples

reviews = reviews.map(preprocess)
restaurant_reviews = pd.concat([restaurant_reviews, reviews], axis = 1)
restaurant_reviews.columns = ['Unnamed: 0.1', 'Unnamed: 0', 'business_id', 'name', 'address', 'city',
       'state', 'stars', 'is_open', 'attributes', 'categories', 'user_id',
       'review_stars', 'text', 'date', 'text_no_symbol', 'preprocessed_text']
restaurant_reviews.to_csv("restaurant_reviews.csv")

##### Detect Bigrams
(x_train, x_test) = train_test_split(reviews, test_size = 0.3)
x_train = x_train.sort_index() # reorder df
x_train = x_train.reset_index() # reset index but preserve old index for later
x_train_reviews = x_train['text']
x_train.to_csv("x_train.csv")

x_test = x_test.sort_index()
x_test = x_test.reset_index()
x_test_reviews = x_test['text']
x_test.to_csv("x_test.csv")

# construct bigrams for entire corpus
start = time.time()
[[re.sub(" ", "_", string) for string in [' '.join((a, b)) for (a, b) in review]] for review in reviews[0:10000].map(bigrams).map(list)]
end = time.time()
print("Run time on 10000 samples: " + str(end - start) + "secs") # 2.9 seconds for 10k docs

bigram_reviews_train = [[re.sub(" ", "_", string) for string in [' '.join((a, b)) for (a, b) in review]] for review in x_train_reviews.map(bigrams).map(list)]
unlist_bigram_reviews_train = list(chain.from_iterable(bigram_reviews_train)) # flatten list

# find most common bigrams
common_bigrams = [a for (a, b) in Counter(unlist_bigram_reviews_train).most_common(100)]
counts = [b for (a, b) in Counter(unlist_bigram_reviews_train).most_common(100)]
np.min(counts) # bigrams show up at least 5022 times in entire corpus


new_tokenized = []

def tokenize_review(review):
    # inpput review
    # ooutput tokenized review accounting for bigrams
    # ie. if tokens are in a common bigram, it will be regarded as a bigram instead of an individual token
    tokenize = []
    j = 0
    while j < len(review) - 1:
        if '_'.join((review[j], review[j+1])) in common_bigrams:
            tokenize.append('_'.join((review[j], review[j + 1])))
            j += 2
        else:
            if len(review[j]) > 3: # only consider words with more than 3 letters
                tokenize.append(review[j])
            j += 1
            if j == len(review) - 1:
                if len(review[j]) > 3:
                    tokenize.append(review[j])
    return tokenize
[tokenize_review(review) for review in reviews[0:2]] # test function

start = time.time()
tokenized_bigram_train = [tokenize_review(review) for review in x_train_reviews]
tokenized_bigram_test = [tokenize_review(review) for review in x_test_reviews]
end = time.time()
print("Run time on 10000 samples: " + str(end - start) + "secs")

x_train["tokenized_bigram_train"] = tokenized_bigram_train
x_test["tokenized_bigram_test"] = tokenized_bigram_test
x_train.to_csv("x_train.csv")
x_test.to_csv("x_test.csv")
# save lists as json

with open("tokenized_bigram_train.json", 'w') as f:
    json.dump(tokenized_bigram_train, f, indent = 2)
with open("tokenized_bigram_test.json", 'w') as f:
    json.dump(tokenized_bigram_test, f, indent=2)

