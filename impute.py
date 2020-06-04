import pandas as pd
import numpy as np
import random
import bayes

def replacer(data, identifier, length):
	dataset = []
	median_set = []

	for i in range(length):
		if(identifier in list(data.iloc[i])):
			pass
		else:
			median_set.append(int(list(data.iloc[i])[6]))

	median = np.median(median_set)

	for i in range(length):
		if(identifier in list(data.iloc[i])):
			pass
		else:
			dataset.append(list(data.iloc[i])[1:6] + [int(list(data.iloc[i])[6])] + list(data.iloc[i])[7:])

	for i in range(length):
		if(identifier in list(data.iloc[i])):
			rep = bayes.call(dataset, list(data.iloc[i])[1:6] + [median] + list(data.iloc[i])[7:])
			point = random.randint(0,len(dataset)-1)
			value = dataset[point][-1]
			while(value != rep):
				point = random.randint(0,len(dataset)-1)
				value = dataset[point][-1]
			return dataset[point][6]
