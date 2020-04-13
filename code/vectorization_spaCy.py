import spacy
from spacy import displacy

spacy.prefer_gpu()
nlp = spacy.load("fr_core_news_sm")

"""
NLP: functions for word embedding with spaCy
	1. Vectorization
	2. Vectors comparison: similiraty and distance 

"""

# vectorization on a complete utterance
def vectorization1(text):
	return nlp(text).vector


# vectorization only on extracted tokens
	#1. average of spaCy vectors
	#2. average of spaCy vectors with TF-IDF (could be vectorization3 )
def vectorization2(tokens):
	return 1/len(tokens)*sum([(nlp(token)).vector for token in tokens])
	#return 1/len(tokens)*sum([token.vector for token in tokens])


# return similarity between two sentences
def similarity(text1, text2):
	return nlp(text1).similarity(nlp(text2))

# return similarity between two tokens
def similarity_tokens(token1, token2):
	return token1.similarity(token2)


#def distance(): # tester d'autre distances dans l'espace sémantique = cosine distance (cf. papier Efficient sentence embedding with...)
#from scipy import spatial
#spatial.distance.cosine(vec1, vec2)
