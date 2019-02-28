from hmmlearn import hmm
import numpy as np
import math
import sys
from pandas import DataFrame,read_csv
import matplotlib.pyplot as plt

plt.style.use('ggplot')

data = read_csv(sys.argv[1])['Close']
diffed = data.diff()
#print(data['Close'])

OBSERVATIONS = (0,1,2,3,4,5)

std = diffed.std()
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

model = hmm.MultinomialHMM(n_components=3)
model.fit(observed)
Z2 = model.predict(observed)

print("Transition Matrix")
print(model.transmat_)
print()

print("Emission Matrix")
print(model.emissionprob_)

bins = 6
for i in range(1):
	plt.hist(mc[i], histtype="step")
	plt.show()
