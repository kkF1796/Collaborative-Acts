"""----------------------------------------------------------------------
	model_tools.py:
		File that contains tools to 
		evaluate the classification

-------------------------------------------------------------------------"""

import pandas as pd
import numpy as np

from sklearn.metrics import cohen_kappa_score 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt


# Function that compute class weights for a given fold
def compute_weights(class_weights, index, imbalance=0):
	weights = sum(class_weights[index])/index.shape[0]
	weights += np.where(weights<1, 1,0)*imbalance
	return dict(enumerate(weights))


# Function that compute class weights for each dyad
def class_weights(y, index):
	weights=[]
	for ind in index:
		weights.append(compute_class_weight('balanced', np.unique(y), y))
	return np.asarray(weights ,dtype=np.float64)


def cross_validation_scores(y_test, y_pred, target):
	# Accuracy
	accuracy_1 = accuracy_score(y_test,y_pred)
	n_true = sum(np.where(y_test==y_pred, 1, 0))
	print(n_true, '/', y_test.shape[0], '=', accuracy_1)

	# Kappa score
	kappa_score_1  = cohen_kappa_score(y_test, y_pred)
	print('Kappa Score: ', kappa_score_1)

	# Confusion matrix
	confusion_mtx_1 = confusion_matrix(y_test, y_pred)
	print('Confusion Matrix: \n', confusion_mtx_1)

	# Classification report
	class_report_1 = classification_report(y_test, y_pred)#, target_names=target)
	print(class_report_1)
	
	return accuracy_1, kappa_score_1



def cross_validation_accuracy(acc_hist):
	print('\nMean Accuracy: ', np.mean(acc_hist))
	print('Min Accuracy: ', min(acc_hist))
	print('Max Accuracy: ', max(acc_hist))

	plt.plot(acc_hist,'m*', label='obtained accuracy')
	plt.axhline(y=0.3,linewidth=0.75, color='r', linestyle='-', label='accuracy==0.3')
	plt.axhline(y=0.4,linewidth=0.75, color='y', linestyle='-', label='accuracy==0.4')
	plt.axhline(y=0.5,linewidth=0.75, color='g', linestyle='-', label='accuracy==0.5')
	for i in range(len(acc_hist)): 
		plt.axvline(x=i,linewidth=0.5, color='c', linestyle='-.')

	plt.legend(loc='upper left')
	plt.xlabel('Fold')
	plt.ylabel('Accuracy')
	plt.title('Accuray computed for each Test according to the different folds')
	plt.show()


def cross_validation_kappa_score(kappa_score_hist):
	print('\nMean Kappa Score: ', np.mean(kappa_score_hist))
	print('Min Kappa Score: ', min(kappa_score_hist))
	print('Max Kappa Score: ', max(kappa_score_hist))

	plt.plot(kappa_score_hist,'m*', label='obtained kappa score')
	plt.axhline(y=0.2,linewidth=0.75, color='r', linestyle='-', label='kappa==0.2')
	plt.axhline(y=0.25,linewidth=0.75, color='y', linestyle='-', label='kappa==0.25')
	plt.axhline(y=0.3,linewidth=0.75, color='g', linestyle='-', label='kappa==0.3')
	for i in range(len(kappa_score_hist)): 
		plt.axvline(x=i,linewidth=0.5, color='c', linestyle='-.')

	plt.legend(loc='upper left')
	plt.xlabel('Fold')
	plt.ylabel('Kappa score')
	plt.title('Kappa score computed for each Test according to the different folds')
	plt.show()

