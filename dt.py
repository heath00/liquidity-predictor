from __future__ import print_function

import os
import subprocess
import csv

# import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz

# from sklearn import tree
import pydotplus 
 
"""
Decision Trees are non-parametric supervised learning method used to for classification and regression. 
scikit-learn uses an optimised version of the CART (Classification and Regression Trees) algorithm to form decision trees.
CART constructs binary trees using the feature and threshold that yield the largest information gain at each node. 
"""

def read_training_data(fname):
	# columns --> 1: SeriousDlqin2yrs, 2: RevolvingUtilizationOfUnsecuredLines, 3: age, 4: NumberOfTime30-59DaysPastDueNotWorse
	#		5: DebtRatio, 6: MonthlyIncome, 7: NumberOfOpenCreditLinesAndLoans, 8: NumberOfTimes90DaysLate, 9: NumberRealEstateLoansOrLines
	#		10: NumberOfTime60-89DaysPastDueNotWorse, 11: NumberOfDependents
	col_start = 1
	col_end = 10
	data = []
	csv_reader = csv.reader(open(fname))
	for line in csv_reader:
	    # append_line = [line[1], line[2], line[3], line[4], line[5], line[6], line[7], line[8], line[9], line[10]]
	    # line[0] is the row label
	    append_line = [line[i] for i in range(col_start,col_end)]
	    # print ("append line is %s" % append_line)
	    data.append(append_line)

	return data

def get_means(data):
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
	print ("sums are %s" % sums)
	print ("entries are %s" % entries)
	means = [sums[i]/float(entries[i]) for i in range(0,9)]
	print ("means are %s" % means)
	return means


def compute_missing_data(data):
	means = get_means(data)
	for d in data:
		for j,x in enumerate(d):
			if x == 'NA':
				d[j] = means[j]
				# print ("setting value to mean of %s" % means[j])
	return data				


def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")



def create_tree(output_file, colored=False):
	"""
	Inputs:
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
	print("headers are %s" % headers)
	data = d[1:]
	data_no_class = [x[1:] for x in data]
	print ("data no class %s" % data_no_class[0])
	classifiers = [x[0] for x in data]
	print ("classifiers %s" % classifiers[0])
	# dt = tree.DecisionTreeClassifier(min_samples_split=20, random_state=99)
	dt = tree.DecisionTreeClassifier()
	dt = dt.fit(data_no_class, classifiers)
	with open("dt.dot", 'w') as f:
		f = tree.export_graphviz(dt, out_file=f)
	os.unlink('dt.dot')
	if not colored:
		dot_data = tree.export_graphviz(dt, out_file=None) 
	else:
		dot_data = tree.export_graphviz(dt, out_file=None, 
	                     feature_names=headers,  
	                     class_names=[str(x) for x in classifiers],  
	                     filled=True, rounded=True,  
	                     special_characters=True)  
	graph = pydotplus.graph_from_dot_data(dot_data) 
	graph.write_pdf(output_file)
	print ("wrote graph to file: %s" % output_file) 





# definitions
train_file = 'data/cs-training.csv'
test_file = 'data/cs-test.csv'

if __name__ == '__main__':
	create_tree("grpahs/dt-colored-new.pdf", colored=True)
