from hmmlearn import hmm
import numpy as np
import math
import sys
from pandas import DataFrame,read_csv
import matplotlib.pyplot as plt

np.random.seed(21)

plt.style.use('ggplot')

N_HIDDEN = 3

data = read_csv(sys.argv[1])['Close']
diffed = data.diff()

OBSERVATIONS = (0,1,2,3,4,5)

std = diffed.std()/2

print("STD == ", diffed.std())
print("Mean == ", diffed.mean())
observed = []
for i in range(len(diffed)):
	if (diffed[i] <= -(std*2)):
		observed.append(0)
	if (-std >= diffed[i] > -(std*2)):
		observed.append(1)
	if (0 >= diffed[i] > -std):
		observed.append(2)

	if (0 <= diffed[i] < std):
		observed.append(3)
	if (std <= diffed[i] < std*2):
		observed.append(4)
	if (std*2 <= diffed[i]):
		observed.append(5)

observed = np.array(observed)

mc = []
for i in range(len(OBSERVATIONS)):
	mc.append([0,0,0,0,0,0])
	for j in range(len(OBSERVATIONS)):
		for k in range(len(observed) - 1):
			if (observed[k] == OBSERVATIONS[i] and observed[k+1] == OBSERVATIONS[j]):
				mc[i][j] += 1

mc = np.array(mc).astype(float)

for i in range(len(mc)):
	total = float(sum(mc[i]))
	for j in range(len(mc[i])):
		mc[i][j] = float(mc[i][j]/total)

print("Simple Markov Chain")
print(mc)
print()

observed = observed.reshape(len(observed), 1)

model = hmm.MultinomialHMM(n_components=N_HIDDEN)
model.fit(observed)
Z2 = model.predict(observed)

print("Transition Matrix")
print(model.transmat_)
print()

print("Emission Matrix")
print(model.emissionprob_)

x_labels = ["", "(-\u221E, -\u03C3), (-2\u03C3, -\u03C3), (-\u03C3, 0), (0, \u03C3), (\u03C3, 2\u03C3), (2\u03C3, \u221E)", "  ", ""]

mc = np.linalg.matrix_power(mc, 6)
## Simple Markov Chain Plots
bins = 6
x_bar = list()
for i in range(6):
	x_bar.append([])
	for j in range(6):
		x_bar[i] += ([j] * int(mc[i][j] * 10000))

fig, ax1 = plt.subplots(6, sharex=True)
plt.suptitle("Simple Markov Chain Probability Matrix")
ax1 = ax1.ravel()
for i in range(6):
	ax1[i].hist(x_bar[i], [0, 1, 2, 3, 4, 5, 6], histtype="stepfilled")
	y_vals = ax1[i].get_yticks()
	ax1[i].set_yticklabels(['{:3.0f}%'.format(x / 100) for x in y_vals])

plt.show()

#### HMM PLOTS
x_bar = list()
for i in range(N_HIDDEN):
	x_bar.append([])
	for j in range(6):
		x_bar[i] += ([j] * int(model.emissionprob_[i][j] * 10000))

model.transmat_ = np.linalg.matrix_power(model.transmat_, 10)
z_bar = list()
for i in range(N_HIDDEN):
	z_bar.append([])
	for j in range(N_HIDDEN):
		z_bar[i] += ([j] * int(model.transmat_[i][j] * 10000))

fig, ax1 = plt.subplots(N_HIDDEN, sharex=True)
plt.suptitle("Transition Probability Distributions (Hidden States Affecting the Market)")
ax1 = ax1.ravel()
for i in range(N_HIDDEN):
	ax1[i].hist(z_bar[i], [i for i in range(N_HIDDEN +1)], histtype="stepfilled")
	y_vals = ax1[i].get_yticks()
	ax1[i].set_yticklabels(['{:3.0f}%'.format(x / 100) for x in y_vals])
	ax1[i].set_xticklabels([])
plt.show()

fig, ax1 = plt.subplots(N_HIDDEN, sharex=True)
plt.suptitle("Emission Probability Distributions (LULU Price Forecasts)")
ax1 = ax1.ravel()
for i in range(N_HIDDEN):
	ax1[i].hist(x_bar[i], [0,1,2,3,4,5,6], histtype="stepfilled")
	y_vals = ax1[i].get_yticks()
	ax1[i].set_yticklabels(['{:3.0f}%'.format(x / 100) for x in y_vals])
	ax1[i].set_xticklabels([])


plt.show()

