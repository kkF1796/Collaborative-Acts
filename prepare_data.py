"""------------------------------------------------------------------------------------
	Preprocessing of the data : 
		File to prepare and save the prepared data with SpaCy

	1) python3 prepare_data.py p ponct spell predict stop lem
		(save the prepared data -obtained tokens for each sentence- in a file)

	2) python3 prepare_data.py v ponct spell predict stop lem
		(save the collaborativeActs object corresponding to 
		the given type of preprocessing)

	3) python3 prepare_data.py f filename  
		(save the collaborativeActs object corresponding to 
		the given type of preprocessing given in the input file)

	ponct: removing of punctuation (0 or 1)
	spell: spell and correction (0 or 1)
	predict: use of most predictive words (0 or given number of words to keep)
	stop: removing of stop words (0 or 1)
	lem: lemmatization (0 or 1)
	tfidf: use of TF-IDF ponderation (0 or 1)
----------------------------------------------------------------------------------------"""
import sys
import pickle

import pandas as pd
import numpy as np

from preprocessing import *
from data_utils import *
from vectorization_spaCy import *

import collaborativeActs


""" Step 0: Get arguments on command line """
arg=sys.argv[1]
v=(arg=='v')
p=(arg=='p')
f=(arg=='f')


ponct=0
spell=0
predict=0
stop=0
lem=0
tfidf=0

if p or v:
	ponct=int(sys.argv[2])
	spell=int(sys.argv[3])
	predict=int(sys.argv[4])
	stop=int(sys.argv[5])
	lem=int(sys.argv[6])
	tfidf=int(sys.argv[7])


filename='collabacts_'+str(ponct)+str(spell)+str(predict)+str(stop)+str(lem)+str(tfidf)+'_cust'#+'.pickle'

if f:
	p_file=sys.argv[2]
	filename='collabacts_prep_data_'#+'.pickle'

print('Type of operation: ',arg)
print('Punctuation removing: ', ponct)
print('Spelling and correction: ', spell)
print('Use of most predictive words: ', predict)
print('Stop words removing: ', stop)
print('Lemmatization: ', lem)
print('TF-IDF: ', tfidf)
print('File name: ', filename,'\n\n')


""" Step 1: Different types of operations """

collabacts=collaborativeActs.CollabActs()
collabacts.init_model('./collaborativeActs.csv')

if p:
	p_file='prep_data_'+str(ponct)+str(spell)+str(predict)+str(stop)+str(lem)
	collabacts.preprocessing(ponct, spell, predict, stop, lem)
	collabacts.save_tokens(p_file)

if v:
	collabacts.preprocessing(ponct, spell, predict, stop, lem)
	collabacts.vectorization(tfidf)

	print(collabacts.vect_data_set[0])


if f:
	collabacts.load_tokens(p_file, preprocess)
	print('LOADED')
	collabacts.vectorization(tfidf)


if v or f:
	print('Writting on file')	
	with open(filename, "wb") as fp:  
		pickle.dump(collabacts, fp)




