"""------------------------------------------------------------------------------------
	SpaCy : 
		Utterance classification in 
		collaborative acts with SpaCy

	python3 spaCy.py preprocess [algo_classif] [n_test]

	preprocess: Type of preprocessing
		0: Complete utterance
		1: Stop words
		2: Stop words + lemmatization
		3: Lemmatization

	algo_classif: chosen algorithm for classification (see classification.py)
	n_test : number of dyades for testing 
----------------------------------------------------------------------------------------"""
import sys
import pickle

import pandas as pd
import numpy as np

#from preprocessing import *
from vectorization_spaCy import *
from data_utils import *

from model_tools import *

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


""" Step 0: Get arguments on command line """
filename=sys.argv[1]

print('Filename: ', filename)


""" Step 1: Extract data from file """
print('Reading file')
with open(filename, "rb") as fp: 
	collabacts = pickle.load(fp)


vect_data_set=collabacts.vect_data_set
labels=collabacts.labels
duration=collabacts.duration
dyads_index=collabacts.dyads_index

collab_acts=collabacts.collab_acts


""" Step 4 (OPTIONAL) : Extra features """
print('Extra Features')
X=np.asarray(vect_data_set,dtype=np.float64)
y=np.asarray(labels,dtype=np.int32)


X = add_feature_time(X, duration, dyads_index)
#X = add_feature_participant(X, participants, dyads_index)


""" Step 5: Classification """
print('Classification')

from sklearn.naive_bayes import GaussianNB

folds = np.asarray(range(len(dyads_index)) ,dtype=np.int32)

kappa_score_hist =[]
acc_hist = []
n_test=1
for n_fold in range(folds.shape[0]-n_test):

	fold_test = np.asarray(range(n_test),dtype=np.int32)+n_fold	
	fold_train = np.setdiff1d(np.array(range(folds.shape[0])), fold_test)

	X_test, y_test = split(X,y, fold_test, dyads_index) 
	X_train, y_train= split(X,y,fold_train, dyads_index)

	print('\nTRAIN: ', fold_train, X_train.shape[0] ,' TEST:', fold_test,  X_test.shape[0])

	#model1=MLPClassifier(hidden_layer_sizes=(240,12), activation='logistic',solver='sgd', alpha=0.001, batch_size='auto', learning_rate='constant', learning_rate_init=0.1, max_iter=1000, verbose=False) 
	model1=SVC(degree=7,class_weight='balanced')
	#model1= GaussianNB()

	#train model 1
	model1.fit(X_train, y_train)# class_weights= , sample_weights=

	#test model 1 
	y_pred_1 = model1.predict(X_test)
	
	accuracy_1, kappa_score_1 = cross_validation_scores(y_test, y_pred_1, collab_acts)

	kappa_score_hist.append(kappa_score_1)
	acc_hist.append(accuracy_1)
	
		
# plot graphe
cross_validation_accuracy(acc_hist)
cross_validation_kappa_score(kappa_score_hist)









