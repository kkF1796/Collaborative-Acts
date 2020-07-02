"""-----------------------------------------------------------
	SpaCy : 
		Utterance classification with SpaCy
		by comparing extracted tokens from sentences
--------------------------------------------------------------"""

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('always')

from preprocessing import *
from vectorization_spaCy import *
from data_utils import *

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.neighbors import KNeighborsClassifier

"""
NLP: class in .csv file
	0. Dyads
	1. Participant
	2. Id
	3. EAT	
	4. StartTime 
	5. EndTime
	6. Duration
	7. Utterances
	8. Subcategories	
	9. Categories
"""

# SCRIPT A REVOIR !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

""" Step 1: Extract data from file """
dataFile='./collaborativeActs.csv'

df = pd.read_csv(dataFile,delimiter="\t",header=None,error_bad_lines=False, encoding="utf8")

#categories, classification of the utterances of the file according to their type
categories = ["Interaction management", "Social relation", "Task management", "Information", "Transactivity", "Tool", "Other","Outside activity"]
ut_by_categ=[[x for x, t in zip (df[7], df[9]) if t == c] for c in categories]

utterances=np.array(df[7])
subcategories=np.array(df[8])
categories=np.array(df[9])

utterances = np.delete(utterances,0)
subcategories = np.delete(subcategories,0)
categories = np.delete(categories,0)

labels = prep_labels(categories, utterances)


X_train,y_train,X_test,y_test = split_data(utterances,labels)


""" Step 2: Preprocessing of data """
utterances_train_ppng=[normalization(utterance) for utterance in X_train]
tokens_train=[tokenization(utterance) for utterance in utterances_train_ppng]
tokens_train=[ delete_stop_words(token) for token in tokens_train]
#tokens=[ delete_punctuation(token) for token in tokens_train]
tokens_train=[ lemmatization(token) for token in tokens_train]


utterances_test_ppng=[normalization(utterance) for utterance in X_test]
tokens_test=[tokenization(utterance) for utterance in utterances_test_ppng]
tokens_test=[ delete_stop_words(token) for token in tokens_test]
#tokens_test=[ delete_punctuation(token) for token in tokens_test]
tokens_test=[ lemmatization(token) for token in tokens_test]


""" Step 3: Vectorization of sentences """
size_vec= 96 # actual size of a word vector

vect_train=[ np.zeros(size_vec) for token in tokens_train]
for i in range(len(tokens_train)):
	if not len(tokens_train[i])==0:
		#print(tokens_train[i])
		vect_train[i]=np.array(vectorization2(tokens_train[i]))


vect_test=[ np.zeros(size_vec) for token in tokens_test]
for i in range(len(tokens_test)):
	if not len(tokens_test[i])==0:
		#print(tokens_test[i])
		vect_test[i]=np.array(vectorization2(tokens_test[i]))


#vect_train=[ vectorization2(token) for token in tokens_train if not len(token)==0]  #### PB à REVOIR : attention à cette étape les [] sont retirés (enlève des sentences) 

#vect_test=[ vectorization2(token) for token in tokens_test ]#if not len(token)==0]  #### PB à REVOIR : attention à cette étape les [] sont retirés (enlève des sentences) 

"""
i=0
for token in tokens_test:
	if(len(token)==0):
		print(utterances[i])
		print(token)
		print(vectorization2(token,size_vec))
	i+=1"""

"""for k in range(len(utterances)):
	print(utterances[k])
	print(tokens_train[k])
	if not len(tokens[k])==0:
		print(vectorization2(tokens[k]))
	print('')"""


""" Step 4: Classification """
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
print('\n',classification_report(np.array(y_test, dtype=int), np.array(y_test_pred,dtype=int), target_names=["Interaction management", "Social relation", "Task management", "Information", "Transactivity", "Tool", "Other","Outside activity"]) )
