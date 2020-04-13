"""-----------------------------------------------------------
	SpaCy : 
		Utterance classification with SpaCy
		by comparing extracted tokens from sentences
--------------------------------------------------------------"""

import pandas as pd
import numpy as np

from preprocessing import *
from vectorization_spaCy import *

"""
NLP: class in .csv file
	0. Dyads
	1. Participant
	2. Id
	3. EAT	
	4. StartTime 
	5. EndTime
	6. Duration
	7. Utterances
	8. Subcategories	
	9. Categories
"""

""" Step 1: Extract data from file """
dataFile='collaborativeActs.csv'

df = pd.read_csv(dataFile,delimiter="\t",header=None,error_bad_lines=False, encoding="utf8")

#categories, classification of the utterances of the file according to their type
categories = ["Interaction management", "Social relation", "Task management", "Information", "Transactivity", "Tool", "Other"]
ut_by_categ=[[x for x, t in zip (df[7], df[9]) if t == c] for c in categories]

utterances=df[7]
subcategories=df[8]
categories=df[9]


""" Step 2: Preprocessing of data """
utterances_ppng=[normalization(utterance) for utterance in utterances]
tokens=[tokenization(utterance) for utterance in utterances_ppng]
tokens=[ delete_stop_words(token) for token in tokens]
#tokens=[ delete_punctuation(token) for token in tokens]
tokens=[ lemmatization(token) for token in tokens]


""" Step 3: Vectorization of sentences """
#vect=[ vectorization2(token) for token in tokens if not len(token)==0]  #### PB à REVOIR : attention à cette étape les [] sont retirés (enlève des sentences) 

"""i=0
for token in tokens:
	if(len(token)==0):
		print(utterances[i])
		print(token)
	i+=1"""

for k in range(len(utterances)):
	print(utterances[k])
	print(tokens[k])
	if not len(tokens[k])==0:
		print(vectorization2(tokens[k]))
	print('')


""" Step 4: Classification """
# it might be possible to use cosine similarity with kNN or clustering but it costs lot of memory
