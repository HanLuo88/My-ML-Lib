import numpy as np
from statistics import mode
import pandas as pd
import math
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
#Python Scikit-Learn natives KNN benutzen, um ein geeignetes K zu ermitteln
#Einlesen der CSV-Datei und Data-Preprocessing
pimaDF = pd.read_csv('diabetes.csv')
testresult_to_int = {'tested_negative': 0, 'tested_positive': 1}
pimaDF['class'] = pimaDF['class'].map(testresult_to_int)

pima_features = pimaDF.iloc[:, :-1]
pima_class = pimaDF.iloc[:,-1]

#Aufteilen der Daten in 4 Untersets
pima_features_train, pima_features_test, pima_class_train, pima_class_test = train_test_split(pima_features, pima_class, test_size=0.2, random_state=15)

###########################################################################################################################

#Knn-Classifier for K up to 300
# accarray = []
# for k in range(1,301):
    
#     pimaKNN = KNeighborsClassifier(n_neighbors=k)
#     #Training
#     pimaKNN.fit(pima_features_train,pima_class_train)
#     #Accuracy
#     acc = pimaKNN.score(pima_features_test,pima_class_test )
#     accarray.append((k,acc))
    #print('K: ', k, 'Accuracy: ', acc)
# print(accarray)
#Beste Accuracy war 78%

###########################################################################################################################

#Jetzt mit n-fold Cross-Validation
acc_mean = []
for k_value in range(1,301):

    pimaCV = KNeighborsClassifier(n_neighbors=k_value)
    #Training mit n-fold
    cv_scores = cross_val_score(pimaCV,pima_features, pima_class, cv=20)
    acc_mean.append(np.mean(cv_scores))
    #print('K: ', k_value,'mean cv_scores: ', np.mean(cv_scores))
print(acc_mean)
print(np.max(acc_mean))
#Beste Accuracy war 75%

###########################################################################################################################

#Jetzt mit XGBoost

