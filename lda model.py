############ Code for creating LDA model for topic modeling ############

########## Source ##########
# https://towardsdatascience.com/6-tips-to-optimize-an-nlp-topic-model-for-interpretability-20742f3047e2

import pandas as pd
import numpy as np
import re
import gensim
from gensim.models import ldamulticore, CoherenceModel
import time
import json
import pprint

x_train = pd.read_csv("reviews_train.csv")
x_test = pd.read_csv("reviews_test.csv")

reviews_train = x_train["tokenized_bigram_train"]
reviews_test = x_test["tokenized_bigram_test"]


# create dictionary
with open("tokenized_bigram_train.json", "r") as f:
    tokenized_bigram_train = json.load(f)

dictionary = gensim.corpora.Dictionary(tokenized_bigram_train)
dictionary.save("id2word")
corpus = [dictionary.doc2bow(text) for text in tokenized_bigram_train]
with open("lda_corpus.json", 'w') as f:
    json.dump(corpus, f, indent = 2) # save corpus

# build base model
start = time.time()
base_lda = gensim.models.ldamodel.LdaModel(corpus, num_topics=4, id2word=dictionary, alpha = "asymmetric", eta = "symmetric",
                                      passes = 10, iterations=100, chunksize=10000)
end = time.time() # ~ one hour
base_lda.print_topics(4)

# tune hyperparameters
num_topics = [4, 5]
alpha = [0.05, 0.3, 0.6, "asymmetric", "symmetric"]
eta = [0.05, 0.3, 0.6, "symmetric"]
i = 1
for topic in num_topics:
    for a in alpha:
        for b in eta:
            lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=topic,
                                                        id2word=dictionary,
                                                        alpha = a,
                                                        eta = b,
                                                        passes = 20,
                                                        chunksize=10000)
            lda_model.save("lda_" + str(topic) + "_" + str(a) + "_" + str(b) + ".model")
            print(str(i), "/40")
            i += 1
            time.sleep(10)
# evaluate model in jupyter
# final model
final_lda_model = gensim.models.ldamodel.LdaModel.load("lda_4_0.05_0.3.model")
topic_number = [0, 1, 2, 3]
topic_words = [["beer", "beach", "atmospher", "wine", "great", "tour", "view", "vibe", "relax", "select", "patio", "brew"],
               ["pizza", "ramen", "taco", "noodl", "sushi", "rice", "thai", "burrito", "broth", "salsa", "mexican", "pork"],
               ["minut", "manag", "tell", "wait", "call", "rude", "horribl", "worst", "custom", "terribl", "poor", "phone"],
               ["bread", "potato", "bacon", "chocol", "salad", "toast", "cake", "pasta", "lobster", "pancak", "waffl", "french"]]
topic_ideas = ["ambiance", "food 1", "service", "food 2"]
topic_num_and_words = pd.concat([pd.DataFrame(topic_number),pd.DataFrame(topic_ideas), pd.DataFrame(topic_words)], axis = 1)
