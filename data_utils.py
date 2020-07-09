import random
import functools as fctl
import numpy as np


# Convert categories into (int) labels
# categories = ["Interaction management", "Social relation", "Task management", "Information", "Transactivity", "Tool", "Other","Outside activity"]
def prep_labels(categories, utterances,collab_acts):
	labels = np.zeros(len(categories))
	for k in range(len(collab_acts)):
		index = np.where(categories == collab_acts[k])
		labels[index]=k
	return labels


# Function that split data into train set according to the given fold 
def split(X,Y,fold, index):
	x = ()
	y = ()

	for n_fold in range(fold.shape[0]):
		x = x + (X[index[fold[n_fold]]], )
		y = y + (Y[index[fold[n_fold]]], )

	X_train = np.concatenate(x, axis=0)
	y_train = np.concatenate(y, axis=0)
	return X_train, y_train


# Function that add time (percentage) of an utterance as a feature (time_utterance / total_time)
def add_feature_time(X, time, index):
	y = np.zeros(X.shape[0])
	for ind in index:
		y[ind] = time[ind] *100 / sum(time[ind]) # time percentage (%)
	return np.insert(X,  X[0].shape[0] ,y, axis=1)
	

# Function that add the type of the previous utterance as a feature
def add_feature_type_last_utt(X, y, index, label):	
	shift = np.r_[label, y[:-1]] # for the first utterance of the dialog, the type of the previous utterance can be 'Other'
	return np.insert(X,  X[0].shape[0] ,shift, axis=1)


#Function that add participants as a feature (pipeline)
	#indicates if the participant who interacts is the last to have interacted
	#ex: -> P1 -> P2 -> P1 -> P1 -> P2.....
		#P1->P2 : 0
		#P1->P1 : 1 
def add_feature_participant(X, y, index):
	z = np.zeros(X.shape[0])
	for ind in index:
		y_ind = y[ind]
		shift = np.r_[0, y[ind][:-1]] 
		z[ind] = np.where( (y_ind == shift)==True, 1,0)
	return np.insert(X,  X[0].shape[0] ,z, axis=1)


# Function that returns exemple of folds 
def get_fold(n):
	#if n==1:
		#fold =
	if n == 2:
		#[ [18, 9, 8], [5, 14], [3, 7], [1, 16], [17, 6], [0, 13], [2, 11], [4, 11], [4, 12], [2, 12], [10, 15]]
		fold = [ [18, 9], [18, 14], [5, 14], [5, 7], [3, 7], [1, 16], [17, 6], [0, 13], [2, 11], [4, 11], [4, 12], [2, 12], [10, 15]]
	if n == 3:
		#[ [18, 11, 13], [5, 12, 6], [3, 15, 7], [1, 10, 14], [17, 4, 9, 8], [17, 2, 9, 8], [0, 2, 16], [0, 4, 16] ]
		#fold = [ [18, 11, 13], [5, 12, 6], [3, 15, 7], [1, 10, 14], [17, 4, 9], [17, 2, 9], [0, 2, 16], [0, 4, 16] ]
		fold = [ [18, 11, 13,8], [5, 12, 6,8], [3, 15, 7,8], [1, 10, 14,8], [17, 4, 9,8], [17, 2, 9,8], [0, 2, 16,8], [0, 4, 16,8] ]
	if n == 5:
		fold = [ [18, 10, 15, 14, 8], [5, 2,12,7,8], [5, 4,12,7,8], [3,0,11,16,9], [1,17,13,6,9]]

	if n == 6:
		fold = [ [18, 10, 15, 14, 8,2], [5, 2,12,7,8,4], [5, 4,12,7,8,2], [3,0,11,16,9,4], [1,17,13,6,9,2]]

	return np.asarray(fold ,dtype=np.int32)


# Find category of a given utterance
def find_categ(utterance, utterances, categories):
	"""for i in range(len(utterances)):
		if(utterance == utterances[i]):
			break;"""
	return categories[np.where(utterances == utterance)]








