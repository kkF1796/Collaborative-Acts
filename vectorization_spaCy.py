"""-------------------------------------------------------------------
	NLP: functions for word embedding with spaCy
		1. Vectorization
		2. Vectors comparison: similiraty and distance 

-------------------------------------------------------------------"""

import numpy as np
import spacy
from spacy import displacy

from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import  f_classif, chi2

spacy.prefer_gpu()
nlp = spacy.load("fr_core_news_sm")

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
	return words[n_values], f_class[n_values]


def remove_empty(tokens):
	size_vec= 96 # actual size of a word vector

	tokens_data_set=[]
	index_data_set = []
	for i in range(len(tokens)):
		if not len(tokens[i])==0:
			index_data_set.append(i)
			tokens_data_set.append(tokens[i])

	return tokens_data_set, index_data_set


def vectorization(tokens, tfidf=0, utterances=None, y=None, most_pred=0):
	size_vec= 96 # actual size of a word vector
	
	if most_pred !=0:
		words, values = most_predictive_words(utterances, y, most_pred)
		#print(list(words),'\n',values, '\n')
		#

	if tfidf:
		vectorizer = TfidfVectorizer()
		X = vectorizer.fit_transform(utterances)
		X= X.toarray()
		words=vectorizer.get_feature_names()

	#txt = input("Type: ")
	vect_data_set=[]
	for i in range(len(tokens)):
		print(i,')')
		if tfidf:
			print('tfidf')
			vect_data_set.append(vectorization3(tokens[i], X[i], words,1))

		if not tfidf and most_pred ==0:
			print('not tfidf')
			vect_data_set.append(vectorization2(tokens[i]))
			#vect_data_set.append(vectorization1( " ".join(tokens[i])) )

		if most_pred!=0:
			print('pond')
			vect_data_set.append(vectorization3(tokens[i], values, list(words),1))
	return vect_data_set


# vectorization on a complete utterance
# (SpaCy already does the sum-average word vectors)
def vectorization1(text):
	#print('\n',text)
	return nlp(text).vector


# vectorization only on extracted tokens
	#1. average of spaCy vectors
def vectorization2(tokens):
	#if flag:
	#print(tokens)	
	return 1/len(tokens)*sum([(nlp(token)).vector for token in tokens if token != ''])
	#return 1/len(tokens)*sum([token.vector for token in tokens])


# vectorization only on extracted tokens
	#2. average of spaCy vectors with TF-IDF (could be vectorization3 )
def vectorization3(tokens, tfidf, words, p=0):
	#N=len(tokens)
	vect=[]
	for token in tokens:
		if token != '':
			#print('\ntoken: ',token)
			pond = p # 0 or 1
			if token in words:
				pond = pond + tfidf[words.index(token)]
				#print('POND: ',tfidf[words.index(token)], pond, token)
			vect.append((nlp(token)).vector * pond)
	N=len(vect)
	return 1/N * sum(vect)
		

# return similarity between two sentences
def similarity(text1, text2):
	return nlp(text1).similarity(nlp(text2))

# return similarity between two tokens
def similarity_tokens(token1, token2):
	return token1.similarity(token2)


def distance(vec1,vec2): 
	return cosine_similarity(vec1,vec2)
	#return spatial.distance.cosine(vec1, vec2)

	#cosine = scipy.spatial.distance.cosine(vector_1, vector_2)
    	#print('Word Embedding method with a cosine distance asses that our two sentences are similar to',round((1-cosine)*100,2),'%')

	#from sklearn.metrics.pairwise import cosine_similarity
