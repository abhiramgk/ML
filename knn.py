import numpy as np
import folds
import random
from math import *


def dist(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)

def neighbour(train, test, number_of_neighbours):
    distances = list()
    for row in train:
        dista = dist(test, row)
        distances.append((row, dista))
    distances.sort(key=lambda t: t[1])
    neighbors = list()
    for i in range(number_of_neighbours):
        neighbors.append(distances[i][0])
    return neighbors

def predict(train, tests, number_of_neighbours):
    predictions=[]
    for i in tests:
        neighbors = neighbour(train,i, number_of_neighbours)
        out = [row[-1] for row in neighbors]
        prediction = max(set(out), key=out.count)
        predictions.append(prediction)
    return predictions

def execute(data, number_of_folds, k):
    dataset = []
    for i in range(len(data)):
        dataset.append(list(data.iloc[i]))
    scores = folds.evalAlgo(dataset, predict, number_of_folds, k)
    print("K-th nearest neighbours: ", np.mean(scores))
    return np.mean(scores)
