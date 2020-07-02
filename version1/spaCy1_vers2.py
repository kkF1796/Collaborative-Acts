"""-----------------------------------------------------------
	SpaCy : 
		Utterance classification with SpaCy
		by comparing extracted tokens from sentences
--------------------------------------------------------------"""

import pandas as pd
import numpy as np

from preprocessing import *
from vectorization_spaCy import *
from data_utils import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

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


""" Step 3: Vectorization of sentences """
vect_data_set = [ vectorization1(utterance) for utterance in utterances_data_set]


""" Step 4 (OPTIONAL) : Add time (percentage) as a feature (time_utterance / total_time)"""


""" Step 5: Classification """
# convert to numy array before split


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


