from math import sqrt, exp, pi
from folds import *
import numpy as np

def separator(data):
	separated = dict()
	for i in range(len(data)):
		column = data[i]
		classValue = column[-1]
		if (classValue not in separated):
			separated[classValue] = list()
		separated[classValue].append(column)
	return separated

def mean(nums):
	return sum(nums)/float(len(nums))

def standardDeviation(nums):
	average = mean(nums)
	variance = sum([(x-average)**2 for x in nums]) / float(len(nums)-1)
	return sqrt(variance)

def summariseData(data):
	summary = [(mean(column), standardDeviation(column), len(column)) for column in zip(*data)]
	del(summary[-1])
	return summary

def summariseByClass(data):
	separated = separator(data)
	summary = dict()
	for classValue, rows in separated.items():
		summary[classValue] = summariseData(rows)
	return summary

def gaussianProbability(x, mean, standardDeviation):
	index = exp(-((x-mean)**2 / (2 * standardDeviation**2 )))
	return (1 / (sqrt(2 * pi) * standardDeviation)) * index

def gaussianProbabilityByClass(summary, row):
	summarySum = sum([summary[label][0][2] for label in summary])
	probabilities = dict()
	for classValue, summaryClass in summary.items():
		probabilities[classValue] = summary[classValue][0][2]/float(summarySum)
		for i in range(len(summaryClass)):
			mean, standardDeviation, dummy = summaryClass[i]
			probabilities[classValue] *= gaussianProbability(row[i], mean, standardDeviation)
	return probabilities

def predict(summary, row):
	probabilities = gaussianProbabilityByClass(summary, row)
	idealValue, idealProbability = None, -1
	for classValue, probability in probabilities.items():
		if idealValue is None or probability > idealProbability:
			idealProbability = probability
			idealValue = classValue
	return idealValue

def call(summary, row):
	model = summariseByClass(summary)
	value = predict(model, row)
	return value

def Bayes(train, test):
	summarize = summariseByClass(train)
	predictions = list()
	for row in test:
		output = predict(summarize, row)
		predictions.append(output)
	return(predictions)

def execute(dataframe, number_of_folds):
	data = []
	for i in range(len(dataframe)):
		data.append(list(dataframe.iloc[i]))
	scores = evalAlgo(data, Bayes, number_of_folds)
	print("Naive Bayes: ",np.mean(scores))
	return np.mean(scores)