"""
Implementation of the knn algorithm for our dataset using the scikit-learn library. Saves graphical output to the graph folder 
For the purpose of this project, we used different K sizes for our implementation. The ouputs can be found in the graph folder 
with each file labeled as 'k'-nn.pdf e.g. 5-nn.pdf for the 5 nearest neighbors output.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets



# n_neighbors = 15

def read_training_data(fname):
	# columns --> 1: SeriousDlqin2yrs, 2: RevolvingUtilizationOfUnsecuredLines, 3: age, 4: NumberOfTime30-59DaysPastDueNotWorse
	#		5: DebtRatio, 6: MonthlyIncome, 7: NumberOfOpenCreditLinesAndLoans, 8: NumberOfTimes90DaysLate, 9: NumberRealEstateLoansOrLines
	#		10: NumberOfTime60-89DaysPastDueNotWorse, 11: NumberOfDependents
	col_start = 1
	col_end = 10
	data = []
	csv_reader = csv.reader(open(fname))
	for line in csv_reader:
	    # line[0] is the row label
	    append_line = [line[i] for i in range(col_start,col_end)]
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



# import some data to play with
# iris = datasets.load_iris()
# X = iris.data[:, :2]  # we only take the first two features. We could
#                       # avoid this ugly slicing by using a two-dim dataset
# y = iris.target

def build_knn_classifer(outfile, n_neighbors):

	#parse data
	d_missing = read_training_data(train_file)
	d = compute_missing_data(d_missing)
	headers = d[0]
	print("headers are %s" % headers)
	data = d[1:]
	data_no_class = [x[1:] for x in data]
	print ("data no class %s" % data_no_class[0])
	classifiers = [x[0] for x in data]
	print ("classifiers %s" % classifiers[0])


	h = .02  # step size in the mesh

	# Create color maps
	cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
	cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

	for weights in ['uniform', 'distance']:
	    # we create an instance of Neighbours Classifier and fit the data.
	    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
	    clf.fit(X, y)

	    # Plot the decision boundary. For that, we will assign a color to each
	    # point in the mesh [x_min, x_max]x[y_min, y_max].
	    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
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
	    plt.title("2-Class classification (k = %i, weights = '%s')"
	              % (n_neighbors, weights))

	plt.show()


# definitions
train_file = 'data/cs-training.csv'
test_file = 'data/cs-test.csv'

if __name__ == '__main__':
	data = read_training_data(train_file)
	build_knn_classifer("grpahs/5-nn.pdf", 5)

