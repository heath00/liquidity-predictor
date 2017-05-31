import csv 


def read_training_data(fname):
	# columns --> 1: SeriousDlqin2yrs, 2: RevolvingUtilizationOfUnsecuredLines, 3: age, 4: NumberOfTime30-59DaysPastDueNotWorse
	#		5: DebtRatio, 6: MonthlyIncome, 7: NumberOfOpenCreditLinesAndLoans, 8: NumberOfTimes90DaysLate, 9: NumberRealEstateLoansOrLines
	#		10: NumberOfTime60-89DaysPastDueNotWorse, 11: NumberOfDependents
	col_start = 1
	col_end = 12
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
	totals = [[] for x in range(0, 11)]
	entries = [0] * 11
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
	means = [sums[i]/float(entries[i]) for i in range(0,11)]
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

d = read_training_data('data/cs-training.csv')	
d_new = compute_missing_data(d)

with open("data/training_no_missing_attrs.csv", "wb") as f:
    writer = csv.writer(f)
    # writer.writerow(headers)
    writer.writerows(d_new)
