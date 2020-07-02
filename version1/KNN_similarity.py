"""---------------------------------------------------
	KNN : 
		KNN algorithm revised with 
		word vector similarity
-------------------------------------------------------"""

import numpy as np

from vectorization_spaCy import *
#import word2vec_model

from sklearn.metrics.pairwise import cosine_similarity

class knn(object):

	def __init__(self):
		pass


	def train(self, X, y):
		self.X_train = X
		self.y_train = y


	def predict(self, X_test, k=1):
		num_test = len(X_test)#.shape[0]
		num_train = len(self.X_train)#.shape[0]
		sim_mtx = np.zeros((num_test, num_train))

		for i in range(num_test):
			for j in range(num_train):
				#print('\n',i,j)
				#sim_mtx = similarity(X_test[i],self.X_train[j])

				sim_mtx = cosine_similarity([X_test[i]],[self.X_train[j]])

				#sim_matrx = word2vec(X_test)
		

		y_pred = np.zeros(num_test)
		for i in range(num_test):
			most_similar_y =  np.argpartition(sim_mtx[i],-k)[-k:]#np.argpartition(dists[i], -k)[:-k]
			most_similar_y_labels = self.y_train[most_similar_y ]

			y_pred[i] = np.argmax(np.bincount(most_similar_y_labels))#max(set(closest_y_labels), key=closest_y_labels.count)

		return y_pred










