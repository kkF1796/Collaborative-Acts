"""---------------------------------------------------
	Word2vec : 
		Build model with sentences
-------------------------------------------------------"""
import os.path
import pandas as pd
import numpy as np

import gensim 
from gensim.models import Word2Vec

from preprocessing import *



class word2vec(object):
	def __init__(self):
		pass

	def init_model(self, model_file='model.bin'):
        	#self.model = Word2Vec.load('model.bin')
		self.model = Word2Vec.load(model_file)

	# build model or train pre-existing one
	def build_model(self, dataFile):
		#dataFile='~/Bureau/TravailPerso/collaborativeActs.csv'
		df = pd.read_csv(dataFile,delimiter="\t",header=None,error_bad_lines=False, encoding="utf8")
		sentences=np.array(df[7])
		sentences=[normalization(sentence) for sentence in sentences]
		sentences=[tokenization(sentence) for sentence in sentences]
		sentences=[from_token_to_text(sentence) for sentence in sentences]
		#print(sentences[2])

		if os.path.exists('model.bin'):
			print("File already exists")
			model = Word2Vec.load('model.bin')
			model.build_vocab(sentences, update=True)
			model.train(sentences, total_examples=len(sentences) , epochs=model.epochs)
			#print('\n',list(model.wv.vocab))
		else:
			model = Word2Vec(sentences, size=10, window=1, min_count=1, workers=1)
			#print('\n',list(model.wv.vocab))
		model.save('model.bin')

	# vectorization only on extracted tokens
		#1. average of spaCy vectors
		#2. average of spaCy vectors with TF-IDF (could be vectorization2 )
	def vectorization(self, tokens):	
		return 1/len(tokens)*sum([(self.model).wv[token] for token in tokens])

	def most_similar(sentence):
		return self.model.wv.most_similar(sentence)


	def similarity(word1, word2):
		return self.model.similarity(word1, word2)






