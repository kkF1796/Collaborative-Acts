"""---------------------------------------------------
	SpaCy : 
		Utterance classification with SpaCy
		by comparing complete sentences
-------------------------------------------------------"""

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('always')

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from preprocessing import *
from vectorization_spaCy import *
from data_utils import *

import KNN_similarity


""" Step 1: Extract data from file """

dataFile='./collaborativeActs.csv'

df = pd.read_csv(dataFile,delimiter="\t",header=None,error_bad_lines=False, encoding="utf8")

# Information about dyads
dyads=np.delete(np.array(df[0]),0)
unique_dyads = np.unique(dyads)
print('Dyads: ', unique_dyads)

# Informations about utterances
utterances = np.delete(np.array(df[7]),0)
subcategories = np.delete(np.array(df[8]),0)
categories = np.delete(np.array(df[9]),0)

collab_acts = np.unique(categories)
print('Categories: ', collab_acts)

# Time scale
duration=np.delete(np.array(df[6]),0)



labels = prep_labels(categories, utterances, collab_acts)

X_train,y_train,X_test,y_test = split_data(utterances,labels) # RETIRER ET TRAVAILLER SUR ALL DATA SET 


""" Step 2: Preprocessing of data """
utterances_train_norm=[normalization(utterance) for utterance in X_train]
utterances_test_norm=[normalization(utterance) for utterance in X_test]


""" Step 3: Vectorization of sentences """
vect_train = [ vectorization1(utterance) for utterance in utterances_train_norm]
vect_test = [ vectorization1(utterance) for utterance in utterances_test_norm]


""" Step 4: Classification """

"""
# Classification: KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10, algorithm='ball_tree', metric='minkowski') # distance computed : euclidean, manhattan, chebyshev, minkowski, wminkowski, seuclidean, mahalanobis  / algorithm : ball_tree, kd_tree
knn.fit(vect_train, y_train);
y_test_pred = knn.predict(vect_test)
print("\nKNN:")
print("Accuray:",accuracy(y_test,y_test_pred))
print("Kappa Score: ",kappa_score(np.array(y_test, dtype=int), np.array(y_test_pred,dtype=int) ))
# convert labels into categories before calculating Kappa score ???

print('\nConfusion Matrix: ')
print('\n',confusion_matrix(np.array(y_test, dtype=int), np.array(y_test_pred,dtype=int)) )
print('\nClassification Report:')
print('\n',classification_report(np.array(y_test, dtype=int), np.array(y_test_pred,dtype=int), target_names=["Interaction management", "Social relation", "Task management", "Information", "Transactivity", "Tool", "Other","Outside activity"]) )"""




















