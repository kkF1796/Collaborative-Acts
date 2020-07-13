"""------------------------------------------------------------------------------------
	SpaCy : 
		Utterance classification in 
		collaborative acts with SpaCy

	python3 spaCy.py filename [algo_classif] [n_test]

	filename: filename for the collaborativeAct objects

	algo_classif: chosen algorithm for classification (see classification.py)
	n_test : number of dyades for testing 
----------------------------------------------------------------------------------------"""
import sys
import pickle

import pandas as pd
import numpy as np

#from preprocessing import *
#from vectorization_spaCy import *
from data_utils import *

from model_tools import *

import collaborativeActs

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

""" Step 0: Get arguments on command line """
filename=sys.argv[1]

algo_classif = 0
n_test = 1

if len(sys.argv) > 2: 
	algo_classif= int(sys.argv[2])

if len(sys.argv) > 3: 
	n_test= int(sys.argv[3])

print('Filename: ', filename)
print('Chosen algorithm for classification: ', algo_classif)
print('Number of dyads for test: ', n_test,'\n\n')


""" Step 1: Extract data from file """
print('Reading file')
with open(filename, "rb") as fp: 
	collabacts = pickle.load(fp)

vect_data_set=collabacts.vect_data_set
labels=collabacts.labels
duration=collabacts.duration
dyads_index=collabacts.dyads_index

collab_acts=collabacts.collab_acts



""" Step 2 (OPTIONAL) : Extra features """
print('Extra Features')
X=np.asarray(vect_data_set,dtype=np.float64)
y=np.asarray(labels,dtype=np.int32)


X = add_feature_time(X, duration, dyads_index)
#X = add_feature_participant(X, participants, dyads_index)


""" Step 3: Classification """
print('Classification')
class_weights = class_weights(y, dyads_index)


n_iter=1
n_dyades=14
for iteration in range(n_iter):

	#Initialize model SVM or NN
	model=MLPClassifier(hidden_layer_sizes=(30), activation='logistic',solver='sgd', alpha=0.001, batch_size='auto', learning_rate='constant', learning_rate_init=0.1, max_iter=1000, verbose=False)
	#model=SVC(degree=7,class_weight='balanced')


	fold_train = np.random.choice(range(len(dyads_index)), n_dyades)
	fold_test = np.setdiff1d(np.array(range(len(dyads_index))), np.array(fold_train))

	print('\n\nTRAIN SET:', fold_train)

	for n_train in range(len(fold_train)):
		fold = np.asarray([fold_train[n_train]],dtype=np.int32)	 
		X_train, y_train= split(X,y,fold, dyads_index)


		weights = compute_weights(class_weights, fold_train,0.0)
		print(n_train,')',fold_train[n_train],'; Weights: ',weights)

		model.fit(X_train, y_train)#class_weights= or sample_weights=

	kappa_score_hist =[]
	acc_hist = []
	i=0
	for n_test in range(fold_test.shape[0]):
		test = np.asarray([fold_test[n_test]],dtype=np.int32)
		X_test, y_test = split(X,y, test, dyads_index)

		print('\n',i,')',' TEST:', fold_test[n_test],  X_test.shape[0])
		i+=1
		" Compute scores "
		#prediction for lstm
		y_pred = model.predict(X_test)

		accuracy, kappa_score = cross_validation_scores(y_test, y_pred, collab_acts)

		kappa_score_hist.append(kappa_score)
		acc_hist.append(accuracy)

	cross_validation_accuracy(acc_hist)
	cross_validation_kappa_score(kappa_score_hist)












