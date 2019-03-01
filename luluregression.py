from pandas import DataFrame,read_csv
from math import sqrt
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt
import numpy as np
import sys

lulu = read_csv(sys.argv[1])
snp = read_csv(sys.argv[2])

END_DATE = '2019-02-22'

def alignData(d1, d2):
	sd = d1
	ld = d2
	if (len(d2) < len(sd)):
		sd = d2
		ld = d1

	startDate = sd['Date'][0]
	
	idx = 0
	for i in range(len(ld)):
		if (ld['Date'][i] == startDate):
			idx = i

	return [np.array([x for x in ld['Close']][idx:]), np.array([x for x in sd['Close']])]

def covariance(x, mean_x, y, mean_y):
	covar = 0.0
	for i in range(len(x)):
		covar += (x[i] - x.mean()) * (y[i] - y.mean()) / (len(x) - 1)
	return covar

def getCoefficients(dataset):
	x = dataset[0]
	y = dataset[1]
	x_mean, y_mean = x.mean(), y.mean()
	b1 = covariance(x, x_mean, y, y_mean) / x.std() ** 2
	b0 = y_mean - b1 * x_mean
	return [b0, b1]


def mse(actual, predicted):
	return ((actual + predicted) ** 2).sum()
	
# Regression
def regression(data):
	train, test = np.array([data[0], data[1]]), np.array([data[0][-100:], data[1][-100:]])
	predictions = list()
	b0, b1 = getCoefficients(data)

	for i in range(len(train[0])):
		yhat = b0 + b1 * train[0][i]
		predictions.append(yhat)

	predictions = np.array(predictions)
	return [train, predictions]


def algo(x, y):
	
	data = np.array([x, y])

	x_mean = x.mean()
	x_std = x.std() ** 2

	y_mean = y.mean()
	y_std = y.std() ** 2

	b0, b1 = getCoefficients(data)
	train, predictions = regression(data)
	rmse = sqrt(mse(train[0], predictions))


	sorted_x = list(x)
	sorted_x.sort()

	print("BETA = " + str(b1))
	plt.plot(x, y, '.')
	plt.plot(sorted_x, [b0 + b1*i for i in sorted_x], '--r')
	plt.show()

x, y = alignData(lulu, snp)
plt.plot(x, y, '.') 
plt.show()

algo(x, y)

x = np.diff(x) / x[:-1]
y = np.diff(y) / y[:-1]

algo(x, y)
