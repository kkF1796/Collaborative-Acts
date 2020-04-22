"""---------------------------------------------------
	SpaCy : 
		Utterance classification with SpaCy
		by comparing complete sentences
-------------------------------------------------------"""

import pandas as pd
import numpy as np

from preprocessing import *
from vectorization_spaCy import *
from data_utils import *

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
utterances_train_norm=[normalization(utterance) for utterance in X_train]
utterances_test_norm=[normalization(utterance) for utterance in X_test]


""" Step 3: Vectorization of sentences """
vect_train = [ vectorization1(utterance) for utterance in utterances_train_norm]
vect_test = [ vectorization1(utterance) for utterance in utterances_test_norm]


""" Step 4: Classification """
# WARNING !!!! Fix SpaCy warning [W007] before this step : a model has to be loaded : https://spacy.io/usage/vectors-similarity
# it might be possible to use .similarity with kNN or clustering but it costs lot of memory

"""ModelsWarning: [W007] The model you're using has no word vectors loaded, so the result of the
Doc.similarity method will be based on the tagger, parser and NER, which may not give useful similarity
judgements. This may happen if you're using one of the small models, e.g. `en_core_web_sm`, which don't ship
with word vectors and only use context-sensitive tensors. You can always add your own word vectors, or use one
of the larger models instead if available"""

"""
# Classification: Random Forest
from sklearn.ensemble import RandomForestRegressor# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(vect_train, y_train);
y_test_pred = rf.predict(vect_test)
print("\nRandom Forest:")
print("Accuray:",accuracy(y_test,y_test_pred))
print("Kappa Score: ",kappa_score(np.array(y_test, dtype=int), np.array(y_test_pred,dtype=int) ))
# convert labels into categories before calculating Kappa score ???
"""

"""
# Classification: Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(vect_train, y_train);
y_test_pred = nb.predict(vect_test)
print("\nNaive Bayes:")
print("Accuray:",accuracy(y_test,y_test_pred))
print("Kappa Score: ",kappa_score(np.array(y_test, dtype=int), np.array(y_test_pred,dtype=int) ))
# convert labels into categories before calculating Kappa score ???
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


"""
# Classification: K-Means clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=8, random_state=0) # distance computed : euclidean, manhattan, chebyshev, minkowski, wminkowski, seuclidean, mahalanobis
kmeans.fit(vect_train);
y_test_pred = kmeans.predict(vect_test) """
"""print("\nK-Means clustering:")
print("Accuray:",accuracy(y_test,y_test_pred))
print("Kappa Score: ",kappa_score(np.array(y_test, dtype=int), np.array(y_test_pred,dtype=int) ))
# convert labels into categories before calculating Kappa score ???"""

"""
# Classification: Agglomerative clustering
from sklearn.cluster import AgglomerativeClustering
agg = AgglomerativeClustering(n_clusters=8) 
assign=agg.fit_predict(vect_train)

clusters=[[x for x, t in zip (X_train, assign) if t == c] for c in range(8)]

for k in clusters[5]:
	print(k)
	print(find_categ(k, utterances, categories))
	print('\n')"""















