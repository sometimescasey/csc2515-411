import warnings

# https://github.com/scikit-learn/scikit-learn/pull/11431/files 
# Suppress warning for now, manually upgrade to 0.21 later
with warnings.catch_warnings():
	warnings.filterwarnings("ignore",category=DeprecationWarning)
	from collections import Mapping, defaultdict 

import numpy as np
import math
from subprocess import call

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import ShuffleSplit


MAX_DEPTH = 20

CLEAN_REAL = "clean_real.txt"
CLEAN_FAKE = "clean_fake.txt"

# Valid sklearn.DecisionTreeClassifier 'criterion' values
CRITERIA = ['gini', 'entropy']

# Class labels
REAL_LABEL = 'R'
FAKE_LABEL = 'F'

# For question 2d): list of splits to try
TRY_WORDS = [
	{ "keyword": "trump", "threshold": 0.035},
	{ "keyword": "donald", "threshold": 0.071 },
	{ "keyword": "hillary", "threshold": 0.045 },
]

def load_data():
	f = open(CLEAN_REAL, "r")
	real_headlines = f.read().splitlines()
	f.close()

	f = open(CLEAN_FAKE, "r")
	fake_headlines = f.read().splitlines()
	f.close()

	count_real = len(real_headlines)
	count_fake = len(fake_headlines)
	count_total = count_real + count_fake

	all_headlines = np.asarray(real_headlines + fake_headlines)

	# vectorizer = CountVectorizer() # Tfidf seems better
	vectorizer = TfidfVectorizer()
	X = vectorizer.fit_transform(all_headlines)

	# Make labels
	real_labels = np.full((count_real, 1), REAL_LABEL)
	fake_labels = np.full((count_fake, 1), FAKE_LABEL)
	all_labels = np.append(real_labels, fake_labels)

	# Append original headline text so we can refer to it later
	a = all_headlines.reshape(1, count_total)
	b = all_labels.reshape(1, count_total)
	y = np.concatenate((a, b), axis=0).T
	# print(y[: ,0]) # text
	# print(y[: ,1]) # labels

	# 70 / 30 split
	X_train, X_temp, y_train, y_temp = custom_train_test_split(X, y, 
		test_size=0.3, random_state=1)

	# split 30 into 15 val, 15 test
	X_val, X_test, y_val, y_test = custom_train_test_split(X_temp, y_temp, 
		test_size=0.5, random_state=1)

	return X_train, X_val, X_test, y_train, y_val, y_test, count_total, vectorizer

def select_model(X_train, y_train, X_val, y_val, max_depth):
	best_index = -1
	best_score = -1
	best_tree = None

	settings = generate_settings(max_depth)
	for setting in settings:
		score, clf = test_settings(setting, X_train, y_train, X_val, y_val)
		if (score > best_score):
			best_score = score
			best_tree = clf

	print("---\nBest hyperparameters: max_depth = {}, criterion = {}, score = {:.4f}"
		.format(
			best_tree.max_depth,
			best_tree.criterion,
			best_score))

	return best_tree
	
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
		criterion=setting["criterion"],
		splitter="random",)
	clf.fit(X=X_train, y=y_train[:, 1]) # train on labels only

	# test on validation set
	y_pred = clf.predict(X=X_val)
	correct = sum(i == j for i, j in zip(y_pred, y_val[:, 1]))
	score = correct / y_val.shape[0]
	print("max_depth: {} | criterion: {} | score: {:.4f}".format(
		str(setting["max_depth"]).ljust(2),
		setting["criterion"].ljust(7),
		score))

	return score, clf

def compute_information_gain(X_train, y_train, vectorizer, keyword, threshold):
	# Split training set based on keyword and <= Tfidf threshold, calculate IG
	# i.e. if tree visualization shows "trump <= 0.035" in first node:
	# keyword = "donald"
	# threshold = 0.035

	feature_names = vectorizer.vocabulary_
	if keyword in feature_names:
		index = feature_names[keyword]
	else:
		print("Cannot calculate IG: Keyword '{}' not found. ".format(keyword))
		return None

	column = X_train[:,index]
	# why not 'column <= threshold'? SparseEfficiencyWarning
	over_threshold = (column > threshold) 
	over_indices = np.nonzero(over_threshold)[0]
	under_indices = np.array([i for i in range(X_train.shape[0]) if i not in over_indices]) # complement
	
	l_child = y_train[: ,1][under_indices]
	r_child = y_train[: ,1][over_indices]
	
	# Parent
	p_n = y_train[: ,1].shape[0]
	p_r = np.count_nonzero(y_train[: ,1] == REAL_LABEL)
	p_f = np.count_nonzero(y_train[: ,1] == FAKE_LABEL)

	# Left child
	l_n = l_child.shape[0]
	l_r = np.count_nonzero(l_child == REAL_LABEL)
	l_f = np.count_nonzero(l_child == FAKE_LABEL)

	# Right child
	r_n = r_child.shape[0]
	r_r = np.count_nonzero(r_child == REAL_LABEL)
	r_f = np.count_nonzero(r_child == FAKE_LABEL)

	# Check that numbers add up, otherwise something is wrong
	assert (p_r + p_f == p_n), \
	"Parent classes do not sum to total: {}:{}, {}:{}, total:{}" \
	.format(
		REAL_LABEL,p_r,
		FAKE_LABEL,p_f,
		p_n)
	assert (l_r + l_f == l_n), \
	"Left child classes do not sum to total: {}:{}, {}:{}, total:{}" \
	.format(
		REAL_LABEL,l_r,
		FAKE_LABEL,l_f,
		l_n)
	assert (r_r + r_f == r_n), \
	"Right child classes do not sum to total: {}:{}, {}:{}, total:{}" \
	.format(
		REAL_LABEL,r_r,
		FAKE_LABEL,r_f,
		r_n)

	p_h = entropy(p_r, p_f)
	l_h = entropy(l_r, l_f)
	r_h = entropy(r_r, r_f)

	ig = p_h - (l_n/p_n * l_h + r_n/p_n * r_h)

	return ig

def entropy(x1, x2):
	# Within a node, define x1 and x2 as the number of items in each category
	x_n = x1+x2
	x_H = -(x1/x_n)*math.log((x1/x_n), 2) + -(x2/x_n)*math.log((x2/x_n), 2)
	return x_H 

def custom_train_test_split(X, y, test_size, random_state):
	# my own implementation of train_test_split, per Piazza @131
	rs = ShuffleSplit(n_splits=1, test_size=test_size, 
		random_state=random_state)
	split = rs.split(X, y)
	
	for train_indices, test_indices in split:
		X_train = X[train_indices]
		X_test = X[test_indices]
		y_train = y[train_indices]
		y_test = y[test_indices]

	return X_train, X_test, y_train, y_test

def visualize(clf, vectorizer):
	export_graphviz(clf, 
		max_depth=2, 
		out_file="tree.dot",
		feature_names = vectorizer.get_feature_names(),
		class_names = clf.classes_,
		)

	# Generate .png
	call(["dot", "-Tpng", "tree.dot", "-o tree.png"])

def question_2d(TRY_WORDS, X_train, y_train, vectorizer):
	print("---")
	print("{} | {} | {}"
			.format("KEYWORD".ljust(10), "Tfidf THRESHOLD", "IG"))
	for word in TRY_WORDS:
		ig = compute_information_gain(X_train, y_train, vectorizer, 
			word['keyword'], word['threshold'])
		if (ig):
			print("{} | {} | {:.4f}"
			.format(word['keyword'].ljust(10), str(word['threshold']).ljust(15), ig))

def main():
	X_train, X_val, X_test, y_train, y_val, y_test, count_total, vectorizer = load_data()
	
	clf = select_model(X_train, y_train, X_val, y_val, MAX_DEPTH)
	
	visualize(clf, vectorizer)
	
	question_2d(TRY_WORDS, X_train, y_train, vectorizer)

if __name__ == "__main__":
	main()
