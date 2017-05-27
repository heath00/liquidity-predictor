import os
import subprocess
import csv

import numpy as np
import pydotplus 
 
# for decision trees
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
# from sklearn import cross_validation
# from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets


# ClASSIFIERS : {DecisionTree, KNN, NaiveBayes, Linear Regression}

def read_training_data(fname):
	# columns --> 1: SeriousDlqin2yrs, 2: RevolvingUtilizationOfUnsecuredLines, 3: age, 4: NumberOfTime30-59DaysPastDueNotWorse
	#		5: DebtRatio, 6: MonthlyIncome, 7: NumberOfOpenCreditLinesAndLoans, 8: NumberOfTimes90DaysLate, 9: NumberRealEstateLoansOrLines
	#		10: NumberOfTime60-89DaysPastDueNotWorse, 11: NumberOfDependents
	col_start = 1
	col_end = 10
	data = []
	f = open(fname)
	csv_reader = csv.reader(f)
	for line in csv_reader:
	    # line[0] is the row label
	    append_line = [line[i] for i in range(col_start,col_end)]
	    data.append(append_line)
	f.close()
	return data

def get_means(data):
	"""
	get the mean values of each columns to fill missing attribute values
	input:
		a list of lists representing the excel data

	returns:
		single list of means associated with each column
	"""
	totals = [[] for x in range(0, 9)]
	entries = [0] * 9
	# data is 2D array -- get each val
	header = 0
	for d in data:
		if header == 0:
			header += 1
			continue
		for i, x in enumerate(d):
			if x != "NA":
				totals[i].append(float(x))
				entries[i] += 1
	
	sums = [sum(total) for total in totals]
	# print ("sums are %s" % sums)
	# print ("entries are %s" % entries)
	means = [sums[i]/float(entries[i]) for i in range(0,9)]
	# print ("means are %s" % means)
	return means


def compute_missing_data(data):
	"""
	fills in the missing attribute with the means of the associated columns 

	inputs:
		a list of lists representing the excel data

	returns:
		the same lists of lists with the missing attributes filled in
	"""
	means = get_means(data)
	for d in data:
		for j,x in enumerate(d):
			if x == 'NA':
				d[j] = means[j]
				# print ("setting value to mean of %s" % means[j])
	return data				



def knn(train_file, k=15):
	"""
	inputs:
		train_file: the file to train and test the classifier


	Returns:
		Nothing
	"""
	n_neighbors = k
	h = .02  # step size in the mesh
		# Create color maps
	cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
	cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

	d_missing = read_training_data(train_file)
	d = compute_missing_data(d_missing)
	headers = d[0]
	# print("headers are %s" % headers)
	data = d[1:]
	x = [i[1:] for i in data] # data with the class attribute missing
	y = [j[0] for j in data] # classifications for the data
	# dt = tree.DecisionTreeClassifier(min_samples_split=20, random_state=99)

	for weights in ['uniform', 'distance']:
	    # we create an instance of Neighbours Classifier and fit the data.
	    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
	    clf.fit(x, y)

	    # Plot the decision boundary. For that, we will assign a color to each
	    # point in the mesh [x_min, x_max]x[y_min, y_max].
	    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
	    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
	    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
	                         np.arange(y_min, y_max, h))
	    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

	    # Put the result into a color plot
	    Z = Z.reshape(xx.shape)
	    plt.figure()
	    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

	    # Plot also the training points
	    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
	    plt.xlim(xx.min(), xx.max())
	    plt.ylim(yy.min(), yy.max())
	    plt.title("3-Class classification (k = %i, weights = '%s')"
	              % (n_neighbors, weights))

	plt.show()

	# clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
	# scores = cross_val_score(estimator=clf, X=x, y=y, cv=10, n_jobs=4)
	# print("KNN 10 Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


def decision_tree(train_file, output_file=None, colored=True, graph=False):
	"""
	Decision Trees are non-parametric supervised learning method used to for classification and regression. 
	scikit-learn uses an optimised version of the CART (Classification and Regression Trees) algorithm to form decision trees.
	CART constructs binary trees using the feature and threshold that yield the largest information gain at each node. 
	This uses an optimized version of CART. 

	** uses means of the column for missing values **
	** perfroms 10-fold cross validation **

	Inputs:
		train_file: the file to train and test the classifier
		output_file: str describing name of the output file you want
		colored: bool descibing the type of output
	reads the training data and accounts for missing values by using the mean of that attribute for all
	training examples (in compute_missing_data). 
	It then builds the decision tree and writes a pdf of the outcome to the output_file

	Returns:
		Nothing
	"""
	d_missing = read_training_data(train_file)
	d = compute_missing_data(d_missing)
	headers = d[0]
	# print("headers are %s" % headers)
	data = d[1:]

	x = [i[1:] for i in data] # data with the class attribute missing
	y = [j[0] for j in data] # classifications for the data
	# dt = tree.DecisionTreeClassifier(min_samples_split=20, random_state=99)

	if graph:
		if output_file is None:
			print("Need to specify an output file to store graph")
			return
		dt = tree.DecisionTreeClassifier()
		dt = dt.fit(x, y)
		with open("dt.dot", 'w') as f:
			f = tree.export_graphviz(dt, out_file=f)
		os.unlink('dt.dot')
		if not colored:
			dot_data = tree.export_graphviz(dt, out_file=None) 
		else:
			dot_data = tree.export_graphviz(dt, out_file=None, 
		                     feature_names=headers,  
		                     class_names=[str(x) for x in classifications],  
		                     filled=True, rounded=True,  
		                     special_characters=True)  
		graph = pydotplus.graph_from_dot_data(dot_data) 
		graph.write_pdf(output_file)
		print ("wrote graph to file: %s" % output_file) 


	clf = tree.DecisionTreeClassifier()
	scores = cross_val_score(estimator=clf, X=x, y=y, cv=10, n_jobs=4)
	print("Decison Tree 10-fold Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


def main():
	train_file = 'data/cs-training.csv'
	decision_tree(train_file)
	knn(train_file, k=10)

if __name__ == '__main__':
	main()



