from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier


def classification(c=0, k=5, layers=(240,12)):
	if c == 0:
		return GaussianNB()

	if c == 1:
		#k = int(np.sqrt(abs(X_train.shape[0]))/3)
		# KNeighborsClassifier(n_neighbors=5, *, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None, **kwargs) 
		# distance computed : euclidean, manhattan, chebyshev, minkowski, wminkowski, seuclidean, mahalanobis
		# p: Power parameter for the Minkowski metric

		return KNeighborsClassifier(n_neighbors=k, weights='uniform', algorithm='ball_tree', metric='minkowski')
	
	if c == 2:
		# MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', *, solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)

		return MLPClassifier(hidden_layer_sizes=layers, activation='logistic',solver='sgd', alpha=0.001, batch_size='auto', learning_rate='constant', learning_rate_init=0.1, max_iter=1000, verbose=False) #96, 240 # (24,24),72,82,120,320

	if c == 3:
		# RandomForestClassifier(n_estimators=100, *, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
		# class_weight{“balanced”, “balanced_subsample”},
		return RandomForestClassifier(class_weight='balanced')

	if c == 4:
		# SVC(*, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)
		# C : Regularization parameter 
		# kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}

		return SVC(degree=7,class_weight='balanced') # deg = 3,5,7,12

	if c == 5:
		# GradientBoostingClassifier(*, loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='deprecated', validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0)

		return GradientBoostingClassifier()




