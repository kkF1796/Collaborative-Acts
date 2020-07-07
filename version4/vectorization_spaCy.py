import numpy as np
import spacy
from spacy import displacy

from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity

spacy.prefer_gpu()
nlp = spacy.load("fr_core_news_sm")

"""
NLP: functions for word embedding with spaCy
	1. Vectorization
	2. Vectors comparison: similiraty and distance 

"""

def vectorization(tokens, preprocess, lemma):
	if not preprocess:
		vect_data_set = [ vectorization1(token) for token in tokens]
		index_data_set = []

	else:
		size_vec= 96 # actual size of a word vector

		index_data_set = []
		vect_data_set=[]
		for i in range(len(tokens)):
			if not len(tokens[i])==0:
				index_data_set.append(i)
				vect_data_set.append(vectorization2(tokens[i], lemma))

	return vect_data_set, index_data_set


# vectorization on a complete utterance
def vectorization1(text):
	return nlp(text).vector


# vectorization only on extracted tokens
	#1. average of spaCy vectors
	#2. average of spaCy vectors with TF-IDF (could be vectorization3 )
def vectorization2(tokens, flag):
	if flag:	
		return 1/len(tokens)*sum([(nlp(token)).vector for token in tokens]) # if lemmatization
	return 1/len(tokens)*sum([token.vector for token in tokens]) # if delete stop-words


# return similarity between two sentences
def similarity(text1, text2):
	return nlp(text1).similarity(nlp(text2))

# return similarity between two tokens
def similarity_tokens(token1, token2):
	return token1.similarity(token2)


def distance(vec1,vec2): # tester d'autre distances dans l'espace sémantique = cosine distance (cf. papier Efficient sentence embedding with...)
	return cosine_similarity(vec1,vec2)
	#return spatial.distance.cosine(vec1, vec2)

	#cosine = scipy.spatial.distance.cosine(vector_1, vector_2)
    	#print('Word Embedding method with a cosine distance asses that our two sentences are similar to',round((1-cosine)*100,2),'%')

	#from sklearn.metrics.pairwise import cosine_similarity
