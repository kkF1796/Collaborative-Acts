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

from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

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
print('Categories: ', collab_acts)

labels = prep_labels(categories, utterances, collab_acts)

# Time scale
duration=np.delete(np.array(df[6]),0)


""" Step 2: Preprocessing of data """
utterances_data_set=[normalization(utterance) for utterance in utterances]
tokens=[tokenization(utterance) for utterance in utterances_data_set]
tokens=[ delete_stop_words(token) for token in tokens]
tokens=[ delete_punctuation(token) for token in tokens]
tokens=[ lemmatization(token) for token in tokens]


""" Step 3: Vectorization of sentences """
size_vec= 96 # actual size of a word vector
"""
tokens=np.array(tokens)
tokens=tokens[np.where(tokens != [])]
labels=labels[np.where(tokens != [])]
#print(tokens[0], type(tokens[0][0]))
#print(np.where(tokens != []))
print(tokens.shape[0])"""

#vect_data_set = [ vectorization2(token) for token in tokens]

vect_data_set=[]
labels_data_set=[]
for i in range(len(tokens)):
	if not len(tokens[i])==0:
		vect_data_set.append(vectorization2(tokens[i]))
		labels_data_set.append(labels[i])

labels=labels_data_set
#print(vect_data_set[3])

"""
vect_data_set=[ np.zeros(size_vec) for token in tokens]
for i in range(len(tokens)):
	if not len(tokens[i])==0:
		#print(tokens_train[i])
		vect_data_set[i]=np.array(vectorization2(tokens[i]))"""


""" Step 4 (OPTIONAL) : Add time (percentage) as a feature (time_utterance / total_time)"""


""" Step 5: Classification """

X=np.asarray(vect_data_set,dtype=np.float64)
X=X[dyads_index[14]]
y=np.asarray(labels,dtype=int32)
y=y[dyads_index[14]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train )
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
print('Accuracy: ',accuracy)


# Naive Bayes, KNN, Neural Network, LSTM

# split data
# Cross validation 
# Test

"""
print("Accuray:",accuracy(y_test,y_test_pred))
print("Kappa Score: ",kappa_score(np.array(y_test, dtype=int), np.array(y_test_pred,dtype=int) ))
print('\nConfusion Matrix: ',confusion_matrix(np.array(y_test, dtype=int), np.array(y_test_pred,dtype=int)) )
print('\nClassification Report:',classification_report(np.array(y_test, dtype=int), np.array(y_test_pred,dtype=int), target_names=["Interaction management", "Social relation", "Task management", "Information", "Transactivity", "Tool", "Other","Outside activity"]) )
"""



