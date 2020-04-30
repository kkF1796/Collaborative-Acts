
    	"""def predict(self, X_test, k=1):#, similarity ='cosine_similarity'):
        	num_test = X_test.shape[0]
        	num_train = self.X_train.shape[0]
		sim_mtx = np.zeros((num_test, num_train))

        	for i in range(num_test):
            		for j in range(num_train):
            			sim_mtx[i,j] = vectorization_spaCy.similarity(X_test[i],self.X_train[j])
             			#sim_mtx[i,j] = cosine_similarity(X_test[i],self.X_train[j])
             			#sim_matrx = word2vec(X_test)

        	num_test = sim_mtx.shape[0]
        	y_pred = np.zeros(num_test)

        	for i in range(num_test):
            		most_similar_y = np.argpartition(dists[i],-k)[-k:]#np.argpartition(dists[i], -k)[:-k] # dists[i,:] # A VERIFIER	
            		most_similar_y_labels = self.y_train[most_similar_y ]

            		y_pred[i]=np.argmax(np.bincount(most_similar_y_labels))#max(set(closest_y_labels), key=closest_y_labels.count)

        	return y_pred	"""
