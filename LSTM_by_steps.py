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

def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		#seq_x, seq_y = sequences[i:end_ix, :-1], sequences[i:end_ix, -1]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)



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


#X = add_feature_time(X, duration, dyads_index)
#X = add_feature_participant(X,collabacts.participants, dyads_index)

#X=add_feature_time_in_dialog(X, duration, dyads_index)
#X=add_feature_n_in_dialog(X, dyads_index)


""" Step 5: Classification """
print('Classification')
class_weights = class_weights(y, dyads_index)

folds = np.asarray(range(len(dyads_index)) ,dtype=np.int32)

kappa_score_hist =[]
acc_hist = []
n_test = 1
for n_fold in range(folds.shape[0]):

	fold_test = np.asarray(range(n_test),dtype=np.int32)+n_fold	
	fold_train = np.setdiff1d(np.array(range(folds.shape[0])), fold_test)

	X_test, y_test = split(X,y, fold_test, dyads_index)
	Z_test = np.insert(X_test,  X_test[0].shape[0] ,y_test, axis=1) 

	X_train, y_train= split(X,y,fold_train, dyads_index)
	Z_train = np.insert(X_train,  X_train[0].shape[0] ,y_train, axis=1) 
	print(Z_train.shape)

	#print('\nTRAIN: ', fold_train, X_train.shape[0] ,' TEST:', fold_test,  X_test.shape[0])

	weights = compute_weights(class_weights, fold_train,0)
	#print('Weights: ',weights)

	n_features = X.shape[1]
	n_steps = 2
	# convert into input/output
	X_train, y_train = split_sequences(Z_train, n_steps)
	X_test, y_test = split_sequences(Z_test, n_steps)
	#print(X_train[0], y_train[0])
	print(X_train.shape, y_train.shape)
	print(X_test.shape, y_test.shape)

	y_train = to_categorical(y_train)
	y_test_categ = to_categorical(y_test)
	#print('\n',y_train)

	hidden_nodes = 95 #95 - 110 #64 # 74- 144 #240
	" Build and Train LSTM "
	#with tf.device('/gpu:0'):
	model = Sequential()
	model.add(LSTM(hidden_nodes,return_sequences=True, input_shape=(n_steps, n_features), name='Layer1')) #return_sequences=True
	model.add(Dropout(0.2))
	model.add(LSTM(hidden_nodes, name='Layer2'))
	model.add(Dropout(0.2))

	model.add(Dense(8, activation='softmax', name='Dense1'))
	#model.build() 
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	#print(model.summary())
	model.fit(X_train, y_train, epochs=60, batch_size=320, verbose=0 ,shuffle=True)#, class_weight=weights) #482

	# Final evaluation of the model
	scores = model.evaluate(X_test, y_test_categ, verbose=0)
	print("Accuracy: %.2f%%" % (scores[1]*100))

	" Compute scores "
	y_pred = model.predict_classes(X_test)

	#print(y_pred)

	accuracy, kappa_score = cross_validation_scores(y_test, y_pred, collab_acts)

	#txt = input("Type: ")

	kappa_score_hist.append(kappa_score)
	acc_hist.append(accuracy)

print(acc_hist)
print('\n',kappa_score_hist)
# plot graphe
cross_validation_accuracy(acc_hist)
cross_validation_kappa_score(kappa_score_hist)














