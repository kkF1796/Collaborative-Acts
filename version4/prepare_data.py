"""------------------------------------------------------------------------------------
	Preprocessing of the data : 
		File to prepare the data with SpaCy

	python3 prepare_data.py preprocess [spell] [ponct]

	preprocess: Type of preprocessing
		0: Complete utterance
		1: Stop words
		2: Stop words + lemmatization
		3: Lemmatization

	spell: indicates the use of spelling and correction (0 or 1)
	ponct: indicates removing of ponctuation
----------------------------------------------------------------------------------------"""
import sys
import pickle

import pandas as pd
import numpy as np

from preprocessing import *
from data_utils import *
from vectorization_spaCy import *


""" Step 0: Get arguments on command line """
preprocess = 0
spell = 1
ponctuation = 1

if len(sys.argv) > 1: 
	preprocess = int(sys.argv[1])

if len(sys.argv) > 2: 
	spell= int(sys.argv[2])

if len(sys.argv) > 3: 
	ponctuation= int(sys.argv[3])

filename='prep_data_'+str(preprocess)+str(spell)+str(ponctuation)+'_custom'#+'.pickle' 

print('Type of preprocessing: ', preprocess)
print('Spelling and correction: ', spell)
print('Removing of ponctuation: ', ponctuation)
print('File name: ', filename,'\n\n')


""" Step 1: Extract data from file """

dataFile='./collaborativeActs.csv'

df = pd.read_csv(dataFile,delimiter="\t",header=None,error_bad_lines=False, encoding="utf8")

utterances = np.delete(np.array(df[7]),0)


""" Step 2: Preprocessing of data """
print('Preprocessing of data')

tokens, lemma=preprocessing(utterances, preprocess,spell, ponctuation)

if preprocess==1:
	tokens=[from_token_to_text(token) for token in tokens]

 
""" Step 3: Saving data """
print('Writting on file')	
with open(filename, "wb") as fp:  
	pickle.dump(tokens, fp)




