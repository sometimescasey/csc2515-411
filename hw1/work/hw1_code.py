import warnings

# conda installed sklearn 0.19.2
# https://github.com/scikit-learn/scikit-learn/pull/11431/files 
# Suppress warning for now, manually upgrade to 0.21 later
with warnings.catch_warnings():
	warnings.filterwarnings("ignore",category=DeprecationWarning)
	from collections import Mapping, defaultdict 

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from subprocess import call


CLEAN_REAL = "clean_real.txt"
CLEAN_FAKE = "clean_fake.txt"

# Valid sklearn.DecisionTreeClassifier 'criterion' values
CRITERIA = ['gini', 'entropy']

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
	real_labels = np.full((count_real, 1), 'R')
	fake_labels = np.full((count_fake, 1), 'F')
	all_labels = np.append(real_labels, fake_labels)

	# Append original headline text so we can refer to it later
	a = all_headlines.reshape(1, count_total)
	b = all_labels.reshape(1, count_total)
	y = np.concatenate((a, b), axis=0).T
	# print(y[: ,0]) # text
	# print(y[: ,1]) # labels

	# 70 / 30 split
	X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=1)

	# split 30 into 15 val, 15 test
	X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.3, random_state=1)

	return X_train, X_val, X_test, y_train, y_val, y_test, count_total, vectorizer

def select_model(X_train, y_train, X_val, y_val, max_depth=5):
	best_index = -1
	best_score = -1
	best_tree = None

	settings = generate_settings(max_depth)
	for setting in settings:
		score, clf = test_settings(setting, X_train, y_train, X_val, y_val)
		if (score > best_score):
			best_score = score
			best_index = settings.index(setting)
			best_tree = clf
			print("new best: " + clf.criterion)

	print("---\nBest hyperparameters: max_depth = {}, criterion = {}, score = {:.4f}"
		.format(
			best_tree.max_depth,
			best_tree.criterion,
			best_score))

	print("about to return: " + best_tree.criterion)
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
		criterion=setting["criterion"])
	clf.fit(X=X_train, y=y_train[:, 1]) # train on labels only

	# test on validation set
	y_pred = clf.predict(X=X_val)
	correct = sum(i == j for i, j in zip(y_pred, y_val[:, 1]))
	score = correct / y_val.shape[0]
	print("max_depth: {} | criterion: {} | score: {:.4f}".format(
		str(setting["max_depth"]).ljust(2),
		setting["criterion"].ljust(7),
		score))

	# Same score behaviour as clf.score
	# print(clf.score(X=X_val, y=y_val[:, 1]))
	print(clf.criterion)
	return score, clf

def compute_information_gain():
	# TODO
	return 0

def predict_sentence(string):
	# TODO: predict for any random headline we grab off the internet
	return 0

def main():
	X_train, X_val, X_test, y_train, y_val, y_test, count_total, vectorizer = load_data()
	clf = select_model(X_train, y_train, X_val, y_val, 20)

	show_gini = (clf.criterion == 'gini')
	print(show_gini)

	print(clf.criterion)

	export_graphviz(clf, 
		max_depth=2, 
		out_file="tree.dot",
		feature_names = vectorizer.get_feature_names(),
		class_names = clf.classes_,
		)

	# Generate .png
	call(["dot", "-Tpng", "tree.dot", "-o tree.png"])


if __name__ == "__main__":
	main()
