import string 

import spacy
from spacy import displacy

from spellchecker import SpellChecker

spacy.prefer_gpu()

nlp = spacy.load("fr_core_news_sm")

spell = SpellChecker(language='fr')

"""
NLP: functions for preprocessing data
	1. Normalization
	2. Tokenization
	3. Deletion of stop words	
	4. Lemmatization
	5. N-grams
"""
import string 

def spell_correction(utterance, punctuation=0):
	#tokens = tokenization(utterance)
	#tokens_text=from_token_to_text(tokens)
	tokens_text = utterance.split(" ")
	if punctuation:
		tokens_text = [token for token in tokens_text if (not token in string.punctuation or token=='?' or token=='!') ] #'(' ')'

	misspelled = spell.unknown(tokens_text)
	for word in misspelled:
		if word in tokens_text:
			index = tokens_text.index(word)
			tokens_text[index] = spell.correction(word)

	spell_correct = " ".join(tokens_text) 
	return spell_correct


def custom_stopwords(utterance):
	custom_list = ["'fin", "s'cours","'scuse","'traument","'ttend","'être", 'alors',"c'est",'de','donc','la','le', 'les','une', 'un','voilà', 'ah','bah',"c'estse","c'ets","c'qu'on","c'qui","c'que","c't'idée","c'tait","c'te", 'ce','ceci','cela','celle', 'celui','ces','cet','cette','ci',"d'acc","d'accus","d'ins","d'jà", "d'la","d'le","d'leur","d'nos",'de','des','du','en','que','quel','quels', 'quelles', 'quelle', 'zyva']
	custom_list.append(['eh','euh','hein','hm','oh', 'tac'])#'ouf','oups','pff' ])
	tokens = utterance.split(" ")
	tokens= [token for token in tokens if not token in custom_list]
	return (" ").join(tokens)


def preprocessing(utterances, preprocess,spell=0, punctuation=0):
	utterance_data_set=[]
	lemma=0
	i = 0
	for utterance in utterances:
		#print(i)
		#i+=1

		tokens=normalization(utterance)

		if spell:
			tokens=spell_correction(tokens, punctuation)

		if preprocess:
			# custom stopwords
			#tokens = custom_stopwords(tokens)		

			tokens=tokenization(tokens)
			#tokens=delete_punctuation(token)

			if preprocess == 1 or preprocess == 2:
				tokens=delete_stop_words(tokens)

			if preprocess == 2 or preprocess == 3:
				tokens=lemmatization(tokens)
				lemma = 1
		utterance_data_set.append(tokens)
	return utterance_data_set, lemma


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
