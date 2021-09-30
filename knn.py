import numpy as np
from statistics import mode
import pandas as pd
import math
from sklearn.model_selection import train_test_split

#Einlesen der Daten
dataset = pd.read_csv('iris_data.csv')
species_to_int = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
dataset['species'] = dataset['species'].map(species_to_int)

#Aufteilen der Daten in Features und Classes
features = dataset.iloc[:, :-1]
species = dataset.iloc[:,-1]


#Aufteilen der Daten in 4 Untersets
features_train, features_test, species_train, species_test = train_test_split(features, species, test_size=0.2, random_state=15)

feat_train_mat = features_train.to_numpy()
feat_test_mat = features_test.to_numpy()
spec_train_mat = species_train.to_numpy()
spec_test_mat = species_test.to_numpy()

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


def findK(k):
    a = []
    for testcounter in range(len(feat_test_mat)):
        a.append(multi_knn(feat_train_mat, spec_train_mat, feat_test_mat[testcounter], k))
    acc = accuracy(a, spec_test_mat)
    return k, acc

for kvalue in range(1,101):
    k, acc = findK(kvalue)
    print('K:: ', k, 'Accuracy: ', acc)

