"""-----------------------------------------------------------
	Word2vec : 
		Utterance classification based on 
		word2vec model
--------------------------------------------------------------"""

import pandas as pd
import numpy as np

import gensim 
from gensim.models import Word2Vec

from preprocessing import *
from vectorization_spaCy import *
from data_utils import *

import word2vec_model

from model_tools import *
from classification import *


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
utterances_data_set=[normalization(utterance) for utterance in utterances]
tokens=[tokenization(utterance) for utterance in utterances_data_set]
tokens=[ delete_stop_words(token) for token in tokens]
#tokens=[ delete_punctuation(token) for token in tokens]
#tokens=[ lemmatization(token) for token in tokens]


""" Step 3: Vectorization of sentences """
tokens=[from_token_to_text(token) for token in tokens]

model=word2vec_model.word2vec()
#model.build_model(dataFile)
model.init_model('model.bin')

#vect_data_set=[ model.vectorization(token) for token in tokens ]#if not len(token)==0]

index_data_set = []
vect_data_set=[]
for i in range(len(tokens)):
	if not len(tokens[i])==0:
		index_data_set.append(i)
		vect_data_set.append(model.vectorization(tokens[i]))


index_data_set=np.asarray(index_data_set,dtype=np.int32)

dyads = dyads[index_data_set]
labels = labels[index_data_set]
participants = participants[index_data_set]
duration = duration[index_data_set]


""" Step 4 (OPTIONAL) : Extra features """
#dyads=np.asarray(dyads_data_set)
unique_dyads = np.unique(dyads)
dyads_index=[np.where(dyads == dyad) for dyad in unique_dyads]

X=np.asarray(vect_data_set,dtype=np.float64)
y=np.asarray(labels,dtype=np.int32)

#X = add_feature_time(X, duration, dyads_index)
#X = add_feature_participant(X, participants, dyads_index)


""" Step 5: Classification """
#n_rand = 12
#fold = np.random.choice(range(len(dyads_index)), n_rand)

folds = np.asarray(range(len(dyads_index)) ,dtype=np.int32)

kappa_score_hist =[]
acc_hist = []
n_test = 1
for n_fold in range(folds.shape[0]-n_test):
	print('\n\n')

	fold_test = np.asarray(range(n_test),dtype=np.int32)+n_fold	
	fold_train = np.setdiff1d(np.array(range(folds.shape[0])), fold_test)

	X_test, y_test = split(X,y, fold_test, dyads_index) 
	X_train, y_train= split(X,y,fold_train, dyads_index)

	print('\nTRAIN: ', fold_train, X_train.shape[0] ,' TEST:', fold_test,  X_test.shape[0])

	k = int(np.sqrt(abs(X_train.shape[0]))/3)
	c = 4
	layers =(240,12)
	model1 = classification(c, k, layers)

	#train model 1 
	model1.fit(X_train, y_train)#, sample_weight=weights) #class_weight

	#test model 1 
	y_pred_1 = model1.predict(X_test)
	
	accuracy_1, kappa_score_1 = cross_validation_scores(y_test, y_pred_1,collab_acts)

	kappa_score_hist.append(kappa_score_1)
	acc_hist.append(accuracy_1)
	
		
# plot graphe
accuracy(acc_hist)
kappa_score(kappa_score_hist)

