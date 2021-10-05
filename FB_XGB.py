import numpy as np
from statistics import mode
import pandas as pd
import math
from sklearn.model_selection import train_test_split


from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot


#Einlesen der Daten
dataset = pd.read_csv('iris_data.csv')
species_to_int = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
dataset['species'] = dataset['species'].map(species_to_int)

#Aufteilen der Daten in Features und Classes
features = dataset.iloc[:, :-1]
species = dataset.iloc[:,-1]


#Aufteilen der Daten in 4 Untersets
features_train, features_test, species_train, species_test = train_test_split(features, species, test_size=0.2, random_state=15)
