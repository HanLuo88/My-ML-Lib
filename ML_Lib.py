import pandas as pd
import numpy as np
import math
from statistics import mode

def euclid_distance(point1, point2):
    sq_sum = 0
    for i in range(len(point1)):
        sq_sum += math.pow(point1[i] - point2[i], 2)
    return math.sqrt(sq_sum)

def multi_knn(trainingsfeatureset, trainingsclassset, testfeatures,  k):
    distance_and_index = []
    k_nearest_classes = []

    for i in range(len(trainingsfeatureset)):
        distance = euclid_distance(trainingsfeatureset[i], testfeatures)
        distance_and_index.append((distance, i))
    
    sorted_distance_and_index = sorted(distance_and_index)
    k_nearest_datapoints = sorted_distance_and_index[:k]
    k_nearest_classes_index = [x[1] for x in k_nearest_datapoints]
    
    for j in range(len(k_nearest_classes_index)):
        k_nearest_classes.append(trainingsclassset[k_nearest_classes_index[j]])
    prediction = mode(k_nearest_classes)
    return prediction

def accuracy(a, b):
    richtig = 0
    falsch = 0
    for c in range(len(a)):
        if a[c] == b[c]:
            richtig += 1
        else:
            falsch += 1
    acc = richtig / (richtig + falsch)
    return acc   
