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

""" Step 1: Extract data from file """
dataFile='~/Bureau/TravailPerso/collaborativeActs.csv'

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
#tokens_train=[ delete_stop_words(token) for token in tokens_train]
#tokens=[ delete_punctuation(token) for token in tokens_train]
#tokens_train=[ lemmatization(token) for token in tokens_train]


utterances_test_ppng=[normalization(utterance) for utterance in X_test]
tokens_test=[tokenization(utterance) for utterance in utterances_test_ppng]
#tokens_test=[ delete_stop_words(token) for token in tokens_test]
#tokens_test=[ delete_punctuation(token) for token in tokens_test]
#tokens_test=[ lemmatization(token) for token in tokens_test]


""" Step 3: Vectorization of sentences """

tokens_train=[from_token_to_text(token) for token in tokens_train]
tokens_test=[from_token_to_text(token) for token in tokens_test]

model=word2vec_model.word2vec()
#model.build_model(dataFile)
model.init_model('model.bin')

vect_train=[ model.vectorization(token) for token in tokens_train ]#if not len(token)==0]
vect_test=[ model.vectorization(token) for token in tokens_test ]#if not len(token)==0]  


""" Step 4: Classification """
# it might be possible to use cosine similarity with kNN or clustering but it costs lot of memory

# Classification: KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10, algorithm='ball_tree', metric='minkowski') # distance computed : euclidean, manhattan, chebyshev, minkowski, wminkowski, seuclidean, mahalanobis  / algorithm : ball_tree, kd_tree
knn.fit(vect_train, y_train);
y_test_pred = knn.predict(vect_test)
print("\nKNN:")
print("Accuray:",accuracy(y_test,y_test_pred))
print("Kappa Score: ",kappa_score(np.array(y_test, dtype=int), np.array(y_test_pred,dtype=int) ))
# convert labels into categories before calculating Kappa score ???
