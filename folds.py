from random import randrange
import fscore

def crossValidator(data, number_of_folds):
	splitData = list()
	copyData = list(data)
	size = int(len(data) / number_of_folds)
	for i in range(number_of_folds):
		fold = list()
		while len(fold) < size:
			index = randrange(len(copyData))
			fold.append(copyData.pop(index))
		splitData.append(fold)
	return splitData

def measureAccuracy(real, predicted):
	correct = 0
	for i in range(len(real)):
		if real[i] == predicted[i]:
			correct += 1
	return correct / float(len(real)) * 100.0
 
def evalAlgo(data, algo, number_of_folds, *args):
	
	folds = crossValidator(data, number_of_folds)
	scores = list()
	fs = list()


	for fold in folds:
		trainingSet = list(folds)
		trainingSet.remove(fold)
		trainingSet = sum(trainingSet, [])
		testingSet = list()

		for row in fold:
			copyRow = list(row)
			testingSet.append(copyRow)
			copyRow[-1] = None

		predicted = algo(trainingSet, testingSet, *args)
		real = [i[-1] for i in fold]
		accuracy = measureAccuracy(real, predicted)
		scores.append(accuracy)

		fs.append(fscore.fscore(real, predicted))
	print("F Score is: ", sum(fs)/ len(fs))
	return scores