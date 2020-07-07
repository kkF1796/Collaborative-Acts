import sys
import pickle

import pandas as pd
import numpy as np

from preprocessing import *
from vectorization_spaCy import *
from data_utils import *

from model_tools import *


import os
os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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


""" Step 0: Get arguments on command line """
filename=sys.argv[1]
preprocess = 0
n_test = 1

if len(sys.argv) > 2: 
	preprocess = int(sys.argv[2])

if len(sys.argv) > 3: 
	n_test= int(sys.argv[3])

print('Filename: ', filename)
print('Type of preprocessing: ', preprocess)
print('Number of dyds for test: ', n_test,'\n\n')


""" Step 1: Extract data from file """

dataFile='./collaborativeActs.csv'

df = pd.read_csv(dataFile,delimiter="\t",header=None,error_bad_lines=False, encoding="utf8")

# Information about dyads
dyads=np.delete(np.array(df[0]),0)
unique_dyads = np.unique(dyads)
print('Dyads: ', unique_dyads)

dyads_index=[np.where(dyads == dyad) for dyad in unique_dyads]

# Informations about utterances
utterances = np.delete(np.array(df[7]),0)
subcategories = np.delete(np.array(df[8]),0)
categories = np.delete(np.array(df[9]),0)

collab_acts = np.unique(categories)
print('Categories: ', collab_acts, '\n')

labels = prep_labels(categories, utterances, collab_acts)

# Participants
participants=np.delete(np.array(df[1]),0)

# Time scale
duration=np.delete(np.array(df[6]),0).astype(np.float)


""" Step 2: Preprocessing of data """
print('Preprocessing of data')

#tokens, lemma=preprocessing(utterances, preprocess,1, 1)

with open(filename, "rb") as fp: 
	tokens = pickle.load(fp)

lemma=1
"""if preprocess:
	data_file=[]
	for data in tokens:
		if data !=[]:
			data = [tokenization(d)[0] for d in data]
		data_file.append(data)
	tokens=data_file"""


""" Step 3: Vectorization of sentences """
print('Vectorization of sentences')
vect_data_set, index_data_set = vectorization(tokens, preprocess, lemma)

if preprocess:
	index_data_set=np.asarray(index_data_set,dtype=np.int32)

	dyads = dyads[index_data_set]
	labels = labels[index_data_set]
	participants = participants[index_data_set]
	duration = duration[index_data_set]

	#dyads=np.asarray(dyads_data_set)
	unique_dyads = np.unique(dyads)
	dyads_index=[np.where(dyads == dyad) for dyad in unique_dyads]


""" Step 4 (OPTIONAL) : Extra features """
print('Extra Features')
X=np.asarray(vect_data_set,dtype=np.float64)
y=np.asarray(labels,dtype=np.int32)


X = add_feature_time(X, duration, dyads_index)


""" Step 5: Classification """
print('Classification')
class_weights = class_weights(y, dyads_index)


folds = np.asarray(range(len(dyads_index)) ,dtype=np.int32)

kappa_score_hist =[]
acc_hist = []
for n_fold in range(folds.shape[0]-n_test):

	fold_test = np.asarray(range(n_test),dtype=np.int32)+n_fold	
	fold_train = np.setdiff1d(np.array(range(folds.shape[0])), fold_test)

	X_test, y_test = split(X,y, fold_test, dyads_index) 
	X_train, y_train= split(X,y,fold_train, dyads_index)

	print('\nTRAIN: ', fold_train, X_train.shape[0] ,' TEST:', fold_test,  X_test.shape[0])

	weights = compute_weights(class_weights, fold_train,0.75)
	print('Weights: ',weights)

	" Prepare standardized data for LSTM"
	n_steps = 1
	n_features = X_train.shape[1]
	X_train = np.reshape(X_train, (X_train.shape[0], n_steps, X_train.shape[1]))
	X_test = np.reshape(X_test, (X_test.shape[0], n_steps, X_test.shape[1]))	
	print(X_train.shape, X_test.shape)

	y_train = to_categorical(y_train)
	y_test_categ = to_categorical(y_test)

	hidden_nodes =240 #64 #240
	" Build and Train LSTM " "A REVOIR POUR LOAD ou GET MODEL"
	with tf.device('/gpu:0'):
		model = Sequential()
		model.add(LSTM(hidden_nodes, return_sequences=True, input_shape=(n_steps, n_features), name='Layer1')) #return_sequences=True
		model.add(Dropout(0.2))
		model.add(LSTM(hidden_nodes, name='Layer2'))
		model.add(Dropout(0.2))
		model.add(Dense(8, activation='softmax', name='Dense1'))

		#model.build() 

		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #optimizer='adam'
		#print(model.summary())
		model.fit(X_train, y_train, epochs=12, batch_size=482, verbose=0 ,shuffle=False, class_weight=weights)

	# Final evaluation of the model
	scores = model.evaluate(X_test, y_test_categ, verbose=0)
	print("Accuracy: %.2f%%" % (scores[1]*100))

	" Compute scores "
	y_pred = model.predict_classes(X_test)

	accuracy, kappa_score = cross_validation_scores(y_test, y_pred, collab_acts)

	kappa_score_hist.append(kappa_score)
	acc_hist.append(accuracy)


print('\n',acc_hist)
print('\n',kappa_score_hist)

# plot graphe
cross_validation_accuracy(acc_hist)
cross_validation_kappa_score(kappa_score_hist)










