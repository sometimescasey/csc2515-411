import warnings

# conda installed sklearn 0.19.2
# https://github.com/scikit-learn/scikit-learn/pull/11431/files 
# Suppress warning for now, manually upgrade to 0.21 later
with warnings.catch_warnings():
	warnings.filterwarnings("ignore",category=DeprecationWarning)
	from collections import Mapping, defaultdict 

import time # kill this
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Constants: file names
CLEAN_REAL = "clean_real.txt"
CLEAN_FAKE = "clean_fake.txt"

# Constants: valid sklearn.DecisionTreeClassifier 'criterion' values
CRITERIA = ['gini', 'entropy']

def load_data():
	f = open(CLEAN_REAL, "r")
	real_headlines = f.read().splitlines()
	f.close()
	print("real_headlines[0]: {}".format(real_headlines[0]))

	f = open(CLEAN_FAKE, "r")
	fake_headlines = f.read().splitlines()
	f.close()
	print("fake_headlines[0]: {}".format(fake_headlines[0]))

	count_real = len(real_headlines)
	count_fake = len(fake_headlines)
	count_total = count_real + count_fake

	print("count_real: {} | count_fake: {}".format(count_real, count_fake))

	all_headlines_temp = real_headlines + fake_headlines
	all_headlines = np.asarray(all_headlines_temp)

	# vectorizer = CountVectorizer() # Tfidf seems better
	vectorizer = TfidfVectorizer()
	X = vectorizer.fit_transform(all_headlines)
	# print("X.shape: {}".format(X.shape))

	# Make labels
	real_y = np.full((count_real, 1), 1) # real headlines get label of 1
	fake_y = np.full((count_fake, 1), 0) # fake headlines get label of 0
	all_y = np.append(real_y, fake_y)

	# Append original headline text so we can refer to it later
	print(all_headlines.shape)
	print(all_y.shape)
	a = all_headlines.reshape(1, count_total)
	b = all_y.reshape(1, count_total)
	y = np.concatenate((a, b), axis=0).T
	print("y.shape: {}".format(y.shape))
	# print(y[: ,0]) # text
	# print(y[: ,1]) # labels
	
	# check we're correct
	print("{} | {}".format(X[count_real-1].toarray(), y[count_real-1]))
	print("{} | {}".format(X[count_real].toarray(), y[count_real]))

	# 70 / 30 split
	X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=1)

	# split 30 into 15 val, 15 test
	X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.3, random_state=1)

	# take a look at the shape of each of these
	print(X_train.shape)
	print(y_train.shape)
	print(X_test.shape)
	print(y_test.shape)

	return X_train, X_val, X_test, y_train, y_val, y_test

def select_model(X_train, y_train, X_val, y_val, max_depth=5):
	best_index = -1
	best_score = -1

	settings = generate_settings(max_depth)
	for setting in settings:
		print("index: {}".format(settings.index(setting)))
		score = test_settings(setting, X_train, y_train, X_val, y_val)
		if (score > best_score):
			best_score = score
			best_index = settings.index(setting)

	print("Best model is: max_depth = {}, criterion = {}, score = {}".format(
		settings[best_index]["max_depth"],
		settings[best_index]["criterion"],
		best_score))
	
def generate_settings(max_depth):
	list = []
	for i in range(1, max_depth+1):
		for criterion in CRITERIA:
			list.append({
				"max_depth": i,
				"criterion": criterion 
			})
	return list

def test_settings(setting, X_train, y_train, X_val, y_val):
	clf = DecisionTreeClassifier(
		max_depth=setting["max_depth"], 
		criterion=setting["criterion"])
	clf.fit(X=X_train, y=y_train[:, 1]) # train on labels only

	# test on validation set
	y_pred = clf.predict(X=X_val)
	correct = sum(i == j for i, j in zip(y_pred, y_val[:, 1]))
	score = correct / y_val.shape[0]
	print("max_depth: {} | criterion: {} | score: {}".format(
		setting["max_depth"],
		setting["criterion"].ljust(7),
		score))

	# Same score behaviour as clf.score
	# print(clf.score(X=X_val, y=y_val[:, 1]))
	return score

def compute_information_gain():
	return 0

def main():
	X_train, X_val, X_test, y_train, y_val, y_test = load_data()
	select_model(X_train, y_train, X_val, y_val, 20)

	


if __name__ == "__main__":
	main()
