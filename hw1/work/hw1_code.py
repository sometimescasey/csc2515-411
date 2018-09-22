import warnings

# conda installed sklearn 0.19.2
# https://github.com/scikit-learn/scikit-learn/pull/11431/files 
# Suppress warning for now, manually upgrade to 0.21 later
with warnings.catch_warnings():
	warnings.filterwarnings("ignore",category=DeprecationWarning)
	from collections import Mapping, defaultdict 

import time # kill this
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


# Constants: file names
CLEAN_REAL = "clean_real.txt"
CLEAN_FAKE = "clean_fake.txt"

def load_data():
	f = open(CLEAN_REAL, "r")
	real_headlines = f.read().splitlines()
	print(real_headlines[0])
	print(real_headlines[1])
	print(real_headlines[2])
	time.sleep(100)
	vectorizer = CountVectorizer()
	real_X = vectorizer.fit_transform(real_headlines)
	real_y = np.full((len(real_X), 1), 1) # real headlines get label of 1

	real_headlines = np.genfromtxt(CLEAN_FAKE, delimiter='\n')
	fake_X = vectorizer.fit_transform(fake_headlines)
	fake_y = np.full((len(real_X), 1), 1) # fake headlines get label of 0

	# Join real and fake headlines into a single set
	X = real_X + fake_X
	y = real_y + fake_y

	print("real_X[0]: {} | real_y[0]".format(real_X[0], real_y[0]))
	

	# 70 / 30 split
	X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=1)

	# split 30 into 15 val, 15 test
	X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.3, random_state=1)

	# take a look at the shape of each of these
	print(X_train.shape)
	print(y_train.shape)
	print(X_test.shape)
	print(y_test.shape)

def select_model():
	return 0

def compute_information_gain():
	return 0

def main():
	load_data()


if __name__ == "__main__":
	main()
