# These are imports and you do not need to modify these.

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd
import sklearn
import random
import math

# ===================================================
# You only need to complete or modify the code below.

def process_data(data, labels):
	"""
	Preprocess a dataset of strings into vector representations.

    Parameters
    ----------
    	data: numpy array
    		An array of N strings.
    	labels: numpy array
    		An array of N integer labels.

    Returns
    -------
    train_X: numpy array
		Array with shape (N, D) of N inputs.
    train_Y:
    	Array with shape (N,) of N labels.
    val_X:
		Array with shape (M, D) of M inputs.
    val_Y:
    	Array with shape (M,) of M labels.
    test_X:
		Array with shape (M, D) of M inputs.
    test_Y:
    	Array with shape (M,) of M labels.
	"""
	
	train_X, temp_X, train_Y, temp_Y = train_test_split(data, labels,
													 test_size = 0.3,
													 random_state = 67,
													 stratify = labels)
	# train_X is 70% of "data" array, train_Y is 70% of "labels" array

	val_X, test_X, val_Y, test_Y = train_test_split(temp_X,
												 temp_Y,
												 test_size = 0.5,
												 random_state = 67,
												 stratify = temp_Y)
	# now we have all train, test, val X and Y

	# Preprocess each dataset of strings into a dataset of feature vectors
	# using the CountVectorizer function. 
	# Note, fit the Vectorizer using the training set only, and then
	# transform the validation and test sets.

	# Initializing CountVectorizer()
	vectorizer = CountVectorizer(
		input = 'content',
		decode_error = 'strict',
		token_pattern = r"(?u)\b\w+\b",
		ngram_range = (1, 2),
		analyzer = 'word'
		)
	
	# Fitting and transforming train X, Y
	train_X = vectorizer.fit_transform(train_X)
	
	# Transforming test and val X
	val_X = vectorizer.transform(val_X)

	test_X = vectorizer.transform(test_X)
	

	# Return the training, validation, and test set inputs and labels

	return(train_X, train_Y, val_X, val_Y, test_X, test_Y)

def select_knn_model(train_X, val_X, train_Y, val_Y):
	"""
	Test k in {1, ..., 20} and return the a k-NN model
	fitted to the training set with the best validation loss.

    Parameters
    ----------
    	train_X: numpy array
    		Array with shape (N, D) of N inputs.
    	train_X: numpy array
    		Array with shape (M, D) of M inputs.
    	train_Y: numpy array
    		Array with shape (N,) of N labels.
    	val_Y: numpy array
    		Array with shape (M,) of M labels.

    Returns
    -------
    best_model : KNeighborsClassifier
    	The best k-NN classifier fit on the training data 
    	and selected according to validation loss.
  	best_k : int
    	The best k value according to validation loss.
	"""
# there are 3266 headlines total 
# 70% of the headlines are in training so ~2286 headlines
# using sqrt(N)/2 we should start with k = 24 (23.906 rounded)
	k = 24
	model_scores = {}
	models = {}
	
	for i in range(1,k+1):
		# fitting model for each k = [1, 24]
		model = KNeighborsClassifier(n_neighbors = i)
		model.fit(train_X, train_Y)
		predicted_vals = model.predict(val_X)

		# evluating model fit
		score = accuracy_score(val_Y, predicted_vals)

		# storing evaluation results
		model_scores[i] = score
		models[i] = model

	# find k with highest accuracy score (same as lowest loss score)
	best_k = max(model_scores, key = model_scores.get)
	best_model = models[best_k]

	return(best_model, best_k)

# You DO NOT need to complete or modify the code below this line.
# ===============================================================


# Set random seed
np.random.seed(3142021)
random.seed(3142021)

def load_data():
	# Load the data
	with open('./clean_fake.txt', 'r') as f:
		fake = [l.strip() for l in f.readlines()]
	with open('./clean_real.txt', 'r') as f:
		real = [l.strip() for l in f.readlines()]

	# Each element is a string, corresponding to a headline
	data = np.array(real + fake)
	labels = np.array([0]*len(real) + [1]*len(fake))
	return data, labels

view_data = load_data()

def main():
	data, labels = load_data()
	train_X, train_Y, val_X, val_Y, test_X, test_Y = process_data(data, labels)

	best_model, best_k = select_knn_model(train_X, val_X, train_Y, val_Y)
	test_accuracy = best_model.score(test_X, test_Y)
	print("Selected K: {}".format(best_k))
	print("Test Acc: {}".format(test_accuracy))


if __name__ == '__main__':
	main()
