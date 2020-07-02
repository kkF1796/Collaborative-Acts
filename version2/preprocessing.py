import spacy
from spacy import displacy

spacy.prefer_gpu()

nlp = spacy.load("fr_core_news_sm")

"""
NLP: functions for preprocessing data
	1. Normalization
	2. Tokenization
	3. Deletion of stop words	
	4. Lemmatization
	5. N-grams
"""

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
