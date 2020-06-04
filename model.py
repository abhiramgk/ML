import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import bayes
import random
import dt
import knn
import impute

pd.options.mode.chained_assignment = None
df =pd.read_table('breast-cancer.data', delimiter=',', names=('id number','clump_thickness','cell_size_uniformity','cell_chape_uniformity','marginal_adhesion','epithelial_cell_size','bare_nuclei','bland_chromatin','normal_nucleoli','mitoses','class'))

for i in df:
	iter = 0
	while(iter < len(list(df[i]))):
		if(df[i][iter] == '?'):
			df[i][iter] = impute.replacer(df,'?', len(df)-1)
		iter += 1

df['bare_nuclei'] = pd.to_numeric(df['bare_nuclei'])

df.drop(['id number'], 1, inplace = True)


label = ["Benign", "Malignant"]
color = []
x_mark = []
count_2 = 0
count_4 = 0
for i in list(df['class']):
	if(i == 2):
		count_2 += 1
	else:
		count_4 += 1

rating = [count_2*100/len(list(df['class'])), count_4*100/len(list(df['class']))]

for i in range(0,len(label)):
	color.append("#" + str(random.randint(10,30)) + str(random.randint(30,60)) + str(random.randint(60,99)))

for i in range(0,2):
	x_mark.append(i + 0.1)
plt.bar(x_mark, rating, color = color, align = "center", tick_label = label, width  = 0.5)
plt.title("Percentage of cancer passes")
plt.ylabel("Percentages")
plt.xlabel("Types")
plt.show()



accuracies_knn = []
accuracies_dt = []
accuracies_nb = []
k_value = []

for i in range(1,7):
	k_value.append(i)
	accuracies_knn.append(knn.execute(df, 6, i))
	accuracies_dt.append(dt.execute(df, 6, i+4, 10))
	accuracies_nb.append(bayes.execute(df, 6))

plt.plot(k_value, accuracies_knn, color='red', marker='.', linestyle='solid')
plt.plot(k_value, accuracies_dt, color='blue', marker='.', linestyle='solid')
plt.plot(k_value, accuracies_nb, color='green', marker='.', linestyle='solid')

plt.legend(['K-NN', 'Decision Tree', 'Naive Bayes'], loc='upper left')
plt.title("Comparison of Models")
plt.xlabel("K values or Max depth or multiple executions")
plt.ylabel("Accuracies")

plt.show()
