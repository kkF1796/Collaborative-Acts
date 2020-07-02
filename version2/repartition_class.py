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
print('Categories: ', collab_acts, '\n')

labels = prep_labels(categories, utterances, collab_acts)

# Participants
participants=np.delete(np.array(df[1]),0)

# Time scale
duration=np.delete(np.array(df[6]),0).astype(np.float)


""" Step 2: Preprocessing of data """
utterances_data_set=[normalization(utterance) for utterance in utterances]


""" Step 3: Vectorization of sentences """
vect_data_set = [ vectorization1(utterance) for utterance in utterances_data_set]


""" Step 4 (OPTIONAL) : Extra features """
X=np.asarray(vect_data_set,dtype=np.float64)
y=np.asarray(labels,dtype=np.int32)


#X = add_feature_time(X, duration, dyads_index)
#X = add_feature_participant(X, participants, dyads_index)




for i in range(len(dyads_index)):
	print('\n\n')
	index = dyads_index[i]
	z = y[index]
	print('Information: ', sum(np.where(z == 0,1,0)))
	print('Interaction management: ', sum(np.where(z == 1,1,0)))
	print('Other: ', sum(np.where(z == 2,1,0)))
	print('Outside activity: ', sum(np.where(z == 2,1,0)))
	print('Social relation: ', sum(np.where(z == 4,1,0)))
	print('Task management: ', sum(np.where(z == 5,1,0)))
	print('Tool: ', sum(np.where(z == 6,1,0)))
	print('Transactivity: ', sum(np.where(z == 7,1,0)))



