import numpy as np
import time
import heapq
import random
from collections import defaultdict


# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT PERFORM ANY FILE IO IN YOUR CODE

# DO NOT CHANGE THE NAME OF THE METHOD my_fit or my_predict BELOW
# IT WILL BE INVOKED BY THE EVALUATION SCRIPT
# CHANGING THE NAME WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, classes to create the Tree, Nodes etc

def initialize_nested_dict():
    return defaultdict(float)

def create_bigrams(word):
    return [''.join(pair) for pair in zip(word[:-1], word[1:])]

def calculate_bigram_freq(words):
    word_count = defaultdict(int)
    bigram_count = defaultdict(lambda: defaultdict(int))
    
    for word in words:
        word_count[word] += 1
        bigrams = create_bigrams(word)
        for bigram in bigrams:
            bigram_count[word][bigram] += 1
    
    return word_count, bigram_count

def train(words):
    word_count, bigram_count = calculate_bigram_freq(words)
    total_words = len(words)
    prior = {word: count / total_words for word, count in word_count.items()}
    likelihood = defaultdict(initialize_nested_dict)
    for word, bigrams in bigram_count.items():
        total_bigrams = sum(bigrams.values())
        for bigram, count in bigrams.items():
            likelihood[word][bigram] = count / total_bigrams
    return prior, likelihood

def predict(bigram_list, prior, likelihood):
    probabilities = []

    for word, prior in prior.items():
        posterior = prior
        for bigram in bigram_list:
            if bigram in likelihood[word]:
                posterior *= likelihood[word][bigram]
            else:
                posterior = 0  
                break

        if posterior > 0:
            if len(probabilities) < 5:
                heapq.heappush(probabilities, (posterior, word))
            else:
                heapq.heappushpop(probabilities, (posterior, word))

    top_5_words = [word for _, word in sorted(probabilities, key=lambda x: -x[0])]    
    refined_words = []
    sorted_given_bigrams = sorted(set(bigram_list))
    for word in top_5_words:
        word_bigrams = sorted(set(create_bigrams(word)))[:5]
        if word_bigrams == sorted_given_bigrams:
            refined_words.append(word)
    return refined_words

################################
# Non Editable Region Starting #
################################
def my_fit( words ):
################################
#  Non Editable Region Ending  #
################################
	# Do not perform any file IO in your code
	# Use this method to train your model using the word list provided
	prior, likelihood = train(words)
	model = (prior, likelihood)
	return model					# Return the trained model


################################
# Non Editable Region Starting #
################################
def my_predict( model, bigram_list ):
################################
#  Non Editable Region Ending  #
################################
	# Do not perform any file IO in your code
	# Use this method to predict on a test bigram_list
	# Ensure that you return a list even if making a single guess
	prior, likelihood = model
	predicted_word = predict(bigram_list, prior, likelihood)
	return predicted_word 					# Return guess(es) as a list
