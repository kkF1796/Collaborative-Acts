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

import matplotlib.pyplot as plt

from sklearn.metrics import cohen_kappa_score 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

def histogram(x, index):
	#x = [value1, value2, value3,....]
	y, x, _ = plt.hist(x, density=True, color = 'magenta',
            edgecolor = 'black')# bins = 8, edgecolor='black')

	#print(y*100)

	plt.xlabel('Collaborative Acts')
	plt.title('Repartition of collaborative Acts for dyad '+str(index))
	plt.show()
	#plt.savefig(index+'_0')
	return y,x


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


#weights = dict(zip(np.unique(y), weights))
#weights = dict(enumerate(weights))
for i in range(len(dyads_index)):
	y = labels[dyads_index[i]]
	weights = compute_class_weight('balanced', np.unique(y), y)
	print('\n',weights)
	histogram(y, i)



