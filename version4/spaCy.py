"""------------------------------------------------------------------------------------
	SpaCy : 
		Utterance classification in 
		collaborative acts with SpaCy

	python3 spaCy.py filename preprocess [algo_classif] [n_test]

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

from preprocessing import *
from vectorization_spaCy import *
from data_utils import *

from model_tools import *
from classification import *


""" Step 0: Get arguments on command line """
filename=sys.argv[1]

preprocess = 0
algo_classif = 0
n_test = 1

if len(sys.argv) > 2: 
	preprocess = int(sys.argv[2])

if len(sys.argv) > 3: 
	algo_classif= int(sys.argv[3])

if len(sys.argv) > 4: 
	n_test= int(sys.argv[4])

print('Filename: ', filename)
print('Type of preprocessing: ', preprocess)
print('Chosen algorithm for classification: ', algo_classif)
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
#X = add_feature_participant(X, participants, dyads_index)


""" Step 5: Classification """
print('Classification')

#fold = np.random.choice(range(len(dyads_index)), 12)

folds = np.asarray(range(len(dyads_index)) ,dtype=np.int32)

kappa_score_hist =[]
acc_hist = []
for n_fold in range(folds.shape[0]-n_test):

	fold_test = np.asarray(range(n_test),dtype=np.int32)+n_fold	
	fold_train = np.setdiff1d(np.array(range(folds.shape[0])), fold_test)

	X_test, y_test = split(X,y, fold_test, dyads_index) 
	X_train, y_train= split(X,y,fold_train, dyads_index)

	print('\nTRAIN: ', fold_train, X_train.shape[0] ,' TEST:', fold_test,  X_test.shape[0])

	model1 = classification(algo_classif, int(np.sqrt(abs(X_train.shape[0]))/3))

	#train model 1 
	model1.fit(X_train, y_train)

	#test model 1 
	y_pred_1 = model1.predict(X_test)
	
	accuracy_1, kappa_score_1 = cross_validation_scores(y_test, y_pred_1, collab_acts)

	kappa_score_hist.append(kappa_score_1)
	acc_hist.append(accuracy_1)
	
		
# plot graphe
cross_validation_accuracy(acc_hist)
cross_validation_kappa_score(kappa_score_hist)









