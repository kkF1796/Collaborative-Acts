"""---------------------------------------------------------
	NLP: functions for preprocessing data
		1. Normalization
		2. Tokenization
		3. Deletion of stop words	
		4. Lemmatization
		5. N-grams
---------------------------------------------------------"""
import string 
import string 
import numpy as np

from sklearn.feature_selection import  f_classif, chi2
from sklearn.feature_extraction.text import CountVectorizer

import spacy
from spacy import displacy

from spellchecker import SpellChecker

spacy.prefer_gpu()

nlp = spacy.load("fr_core_news_sm")

spell = SpellChecker(language='fr')



def preprocessing(utterances, y, ponct=0, spell=0, grammar=0, stop=0, lem=0):
	utterance_data_set=[]
	"""if predict != 0: # 0 or value of the n most predicitve values
		most_pred = most_predictive_words(utterances, y, predict)"""
	print('PREPROCESS: ')
	for utterance in utterances:
		print('\n\n',utterance)

		#tokens=normalization(utterance)
		tokens=tokenization(utterance)
		
		if grammar:
			tokens=[token for token in tokens if token.pos_ != 'DET' and token.pos_ != 'CONJ' and token.pos_ != 'SCONJ']# and token.pos_ != 'ADP']

		if stop:
			tokens=delete_stop_words(tokens)
		if lem:
			tokens=lemmatization(tokens)
		
		if not lem:
			tokens=[token.text for token in tokens]

		if ponct:
			tokens= [token for token in tokens if (not token in string.punctuation or token=='?' or token=='!') ] #not token.isnumeric() or

		print(tokens)
		#tokens=list(set(tokens))
		#txt = input("Type: ")
		utterance_data_set.append(tokens)
	return utterance_data_set


# function that removes upper cases from a text
def normalization(text):
	return text.lower()


# function that returns tokens from a text
def tokenization(text):
	doc = nlp(text)
	return [token for token in doc]


# function that returns text from tokens
def from_token_to_text(tokens):
	return [token.text for token in tokens]

	
# function that delete stopwords and return tokens from a text	
def delete_stop_words(tokens):
	return [token for token in tokens if not token.is_stop]


# function that delete punctuation and return tokens from a text
def delete_punctuation(tokens):
	return [token for token in tokens if not token.is_punct]
	#return [token for token in tokens if ((not token.is_punct or token.text == '!' or token.text =='?')) ]


# function that return lemmas from tokens
def lemmatization(tokens):
	return [token.lemma_ for token in tokens]


# function that tokens ~ n-grams
"""def n_grams(text):
	return 0"""

# function that returns labels from a text
def labels(text):
	doc = nlp(text)
	return [(token.text, token.label_) for token in doc.ents]
