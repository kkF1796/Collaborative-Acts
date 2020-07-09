"""------------------------------------------------------------------------------------
	LSTM : 
		Utterance classification in 
		collaborative acts with LSTM

	python3 lstm.py filename 


----------------------------------------------------------------------------------------"""
import sys
import pickle

import os
#os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

import tensorflow as tf
from tensorflow import keras

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.models import load_model

#import LSTM_model

import pandas as pd
import numpy as np

from data_utils import *
from model_tools import *


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


""" Step 2 (OPTIONAL) : Extra features """
print('Extra Features')
X=np.asarray(vect_data_set,dtype=np.float64)
y=np.asarray(labels,dtype=np.int32)


#X = add_feature_time(X, duration, dyads_index)


""" Step 3: Classification """
print('Classification')
class_weights = class_weights(y, dyads_index)


n_steps = 1
n_features = X.shape[1]
hidden_nodes =144#64 #240

n_iter=1
n_dyades=7
for iteration in range(n_iter):


	#tf.keras.backend.clear_session()

	#with tf.device('/gpu:0'):
	model = Sequential()
	model.add(LSTM(hidden_nodes, return_sequences=True,input_shape=(n_steps, n_features), name='Layer1')) #return_sequences=True
	model.add(Dropout(0.5))

	model.add(LSTM(hidden_nodes, name='Layer2'))
	model.add(Dropout(0.5))

	model.add(Dense(8, activation='softmax', name='Dense1'))
	#model.build() 
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	#print(model.summary())


	fold_train = np.random.choice(range(len(dyads_index)), n_dyades)
	fold_test = np.setdiff1d(np.array(range(len(dyads_index))), np.array(fold_train))

	print('\n\nTRAIN SET:', fold_train)

	#model = load_model('lstm_model.h5')

	for n_train in range(len(fold_train)):
		fold = np.asarray([fold_train[n_train]],dtype=np.int32)	 
		X_train, y_train= split(X,y,fold, dyads_index)

		# standardize data set for lstm
		X_train = np.reshape(X_train, (X_train.shape[0], n_steps, X_train.shape[1]))
		y_train = to_categorical(y_train)

		weights = compute_weights(class_weights, fold_train,0.0)
		print(n_train,')',fold_train[n_train],'; Weights: ',weights)

		#n_epochs=int(X_train.shape[0]/3)
		model.fit(X_train, y_train, epochs=200, batch_size=X_train.shape[0], verbose=1 ,shuffle=False, class_weight=weights)#X_train.shape[0]

		scores = model.evaluate(X_train, y_train, verbose=0)
		print("Accuracy: %.2f%%" % (scores[1]*100))


	kappa_score_hist =[]
	acc_hist = []
	i=0
	for n_test in range(fold_test.shape[0]):

		test = np.asarray([fold_test[n_test]],dtype=np.int32)
		X_test, y_test = split(X,y, test, dyads_index)

		# standardize data set for lstm
		X_test = np.reshape(X_test, (X_test.shape[0], n_steps, X_test.shape[1]))

		print('\n',i,')',' TEST:', fold_test[n_test],  X_test.shape[0])
		i+=1
		" Compute scores "
		#prediction for lstm
		y_pred = model.predict_classes(X_test)

		accuracy, kappa_score = cross_validation_scores(y_test, y_pred, collab_acts)

		kappa_score_hist.append(kappa_score)
		acc_hist.append(accuracy)

	cross_validation_accuracy(acc_hist)
	cross_validation_kappa_score(kappa_score_hist)

	#model.save(filename)














