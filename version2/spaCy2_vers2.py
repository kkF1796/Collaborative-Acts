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

import matplotlib.pyplot as plt

from sklearn.metrics import cohen_kappa_score 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

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
print('Categories: ', collab_acts,'\n')

labels = prep_labels(categories, utterances, collab_acts)

# Participants
participants=np.delete(np.array(df[1]),0)

# Time scale
duration=np.delete(np.array(df[6]),0).astype(np.float)


""" Step 2: Preprocessing of data """
utterances_data_set=[normalization(utterance) for utterance in utterances]
tokens=[tokenization(utterance) for utterance in utterances_data_set]
#tokens=[ delete_stop_words(token) for token in tokens]
#tokens=[ delete_punctuation(token) for token in tokens]
#tokens=[ lemmatization(token) for token in tokens]


""" Step 3: Vectorization of sentences """
size_vec= 96 # actual size of a word vector

index_data_set = []
vect_data_set=[]
for i in range(len(tokens)):
	if not len(tokens[i])==0:
		index_data_set.append(i)
		vect_data_set.append(vectorization2(tokens[i]))


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

#folds = np.asarray([1,5,7,9,12,17] ,dtype=np.int32)

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
	c = 3
	layers =(240,12)
	model1 = classification(c, k, layers)

	#train model 1 
	model1.fit(X_train, y_train)#, sample_weight=weights) #class_weight

	#test model 1 
	y_pred_1 = model1.predict(X_test)
	
	# Accuracy
	accuracy_1 = accuracy_score(y_test,y_pred_1)
	n_true = sum(np.where(y_test==y_pred_1, 1, 0))
	print(n_true, '/', X_test.shape[0], '=', accuracy_1)

	# Kappa score
	kappa_score_1  = cohen_kappa_score(y_test, y_pred_1)
	print('Kappa Score: ', kappa_score_1)

	# Confusion matrix
	confusion_mtx_1 = confusion_matrix(y_test, y_pred_1)
	print('Confusion Matrix: \n', confusion_mtx_1)

	# Classification report
	class_report_1 = classification_report(y_test, y_pred_1, target_names=collab_acts)


	kappa_score_hist.append(kappa_score_1)
	acc_hist.append(accuracy_1)
	
		
# plot graphe
print('\nMean Accuracy: ', np.mean(acc_hist))
print('Min Accuracy: ', min(acc_hist))
print('Max Accuracy: ', max(acc_hist))

print('\nMean Kappa Score: ', np.mean(kappa_score_hist))
print('Min Kappa Score: ', min(kappa_score_hist))
print('Max Kappa Score: ', max(kappa_score_hist))

plt.plot(acc_hist,'m*')
plt.axhline(y=0.3,linewidth=0.75, color='r', linestyle='-')
plt.axhline(y=0.4,linewidth=0.75, color='y', linestyle='-')
plt.axhline(y=0.5,linewidth=0.75, color='g', linestyle='-')
for i in range(len(acc_hist)): 
	plt.axvline(x=i,linewidth=0.5, color='c', linestyle='-.')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Accuray computed for each Test according to the different folds')
plt.show()

plt.plot(kappa_score_hist,'m*')
plt.axhline(y=0.2,linewidth=0.75, color='r', linestyle='-')
plt.axhline(y=0.25,linewidth=0.75, color='y', linestyle='-')
plt.axhline(y=0.3,linewidth=0.75, color='g', linestyle='-')
for i in range(len(kappa_score_hist)): 
	plt.axvline(x=i,linewidth=0.5, color='c', linestyle='-.')
plt.xlabel('Fold')
plt.ylabel('Kappa score')
plt.title('Kappa score computed for each Test according to the different folds')
plt.show()
