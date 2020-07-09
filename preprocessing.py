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


def most_predictive_words(utterances, y, n):
	vectorizer =CountVectorizer()
	X = vectorizer.fit_transform(utterances)
	words = np.array(vectorizer.get_feature_names())
	X = X.toarray()
	if n==1:
		k = int(X.shape[1])
	else:
		k=n
	f_class, p_val= f_classif(X, y)
	n_values=np.array(f_class.argsort()[-k:][::-1])
	return words[n_values]

def spell_correction(tokens):
	#tokens_text = utterance.split(" ")
	tokens_text=tokens
	misspelled = spell.unknown(tokens_text)
	for word in misspelled:
		if word in tokens_text:
			index = tokens_text.index(word)
			tokens_text[index] = spell.correction(word)

	#spell_correct = " ".join(tokens_text) 
	return tokens_text #spell_correct



def preprocessing(utterances, y, ponct=0, spell=0, predict=0, stop=0, lem=0):
	utterance_data_set=[]
	if predict != 0: # 0 or value of the n most predicitve values
		most_pred = most_predictive_words(utterances, y, predict)
	for utterance in utterances:
		#print('\n',utterance)

		tokens=normalization(utterance)
		tokens=tokens.split(" ")

		if predict != 0:
			tokens=[token for token in tokens if token in most_pred]
		if ponct:
			tokens= [token for token in tokens if (not token in string.punctuation or token=='?' or token=='!') ]
		if spell:
			tokens=spell_correction(tokens)

		if stop or lem:
			tokens=" ".join(tokens)
			tokens=tokenization(tokens)
			if stop:
				tokens=delete_stop_words(tokens)
				if not lem:
					tokens=[token.text for token in tokens]
			if lem:
				tokens=lemmatization(tokens)

		#print(tokens)
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
