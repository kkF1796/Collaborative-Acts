import random
import functools as fctl
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html

"""
Classification: supervised or unsupervised algorithm 
	1. 
	2. 
	3. 

"""

# Find category of a given utterance
def find_categ(utterance, utterances, categories):
	for i in range(len(utterances)):
		if(utterance == utterances[i]):
			break;
	return categories[i]


# Convert categories into (float) labels
# categories = ["Interaction management", "Social relation", "Task management", "Information", "Transactivity", "Tool", "Other","Outside activity"]
def prep_labels(categories, utterances):
	labels = np.zeros(len(categories))
	for k in range(len(categories)):
		if(categories[k] == 'Interaction management'):
			labels[k]=0
		elif (categories[k] == 'Social relation'):
			labels[k]=1
		elif (categories[k] ==  'Task management'):
			labels[k]=2
		elif (categories[k] == 'Information'):
			labels[k]=3
		elif(categories[k] == 'Transactivity'):
			labels[k]=4
		elif(categories[k] == 'Tool'):
			labels[k]=5
		elif(categories[k] == 'Other'):
			labels[k]=6
		elif(categories[k] == 'Outside activity'):
			labels[k]=7
		else:
			print("Unrecognized category for utterance ",k, categories[k],", ",utterances[k])
	return labels



# Data splitting
def split_data(X,y):

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0) #42

	"""print('X_train shape:', X_train.shape)
	print('y_train shape:', y_train.shape)
	print('X_test shape:', X_test.shape)
	print('y_test shape:', y_test.shape)"""

	return X_train,y_train,X_test,y_test



# Compute accuracy
def accuracy(y_test,y_test_pred):
	num_correct_test = float(np.sum(y_test_pred == y_test))
	accuracy_test = float(np.sum(y_test_pred == y_test) / len(y_test) ) * 100
	#print('Test: Got %d / %d correct => accuracy: %f' % (num_correct_test, len(y_test), accuracy_test))
	return accuracy_test


# Compute Kappa score
def kappa_score(y, y_pred):
	return cohen_kappa_score(y, y_pred, labels=None, weights=None) # None means no weighted; “linear” means linear weighted; “quadratic” means quadratic weighted

	#Y_pred = new_model.predict(X_test_dtm)
	#cohen_score = cohen_kappa_score(Y_test, Y_pred)





