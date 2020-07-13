import pickle

import pandas as pd
import numpy as np

from preprocessing import *
from vectorization_spaCy import *
from data_utils import *


class CollabActs(object):
	def __init__(self):
		pass

	def init_model(self, dataFile):
		df = pd.read_csv(dataFile,delimiter="\t",header=None,error_bad_lines=False, encoding="utf8")

		# Information about dyads
		self.dyads=np.delete(np.array(df[0]),0)
		unique_dyads = np.unique(self.dyads)

		self.dyads_index=[np.where(self.dyads == dyad) for dyad in unique_dyads]

		# Informations about utterances
		self.utterances = np.delete(np.array(df[7]),0)
		#subcategories = np.delete(np.array(df[8]),0)
		self.categories = np.delete(np.array(df[9]),0)

		self.collab_acts = np.unique(self.categories)
		#print('Categories: ', collab_acts, '\n')

		self.labels = prep_labels(self.categories, self.utterances, self.collab_acts)

		# Participants
		self.participants=np.delete(np.array(df[1]),0)

		# Time scale
		self.duration=np.delete(np.array(df[6]),0).astype(np.float)

	def preprocessing(self,ponct=0, spell=0, predict=0, stop=0, lem=0):
		tokens = []
		for i in range(len(self.dyads_index)):
			index=self.dyads_index[i]
			tokens = tokens + preprocessing(self.utterances[index], self.labels[index], ponct, spell, predict, stop, lem)

		self.tokens=tokens


	def load_tokens(self,filename):
		with open(filename, "rb") as fp: 
				self.tokens = pickle.load(fp)
		#self.lemma=1


	def save_tokens(self,filename='prep_data'):
		with open(filename, "wb") as fp:  
			pickle.dump(self.tokens, fp)

	def set_index(self, index_data_set):
		index_data_set=np.asarray(index_data_set,dtype=np.int32)

		self.utterances=self.utterances[index_data_set]

		self.dyads = self.dyads[index_data_set]
		self.labels = self.labels[index_data_set]
		self.participants = self.participants[index_data_set]
		self.duration = self.duration[index_data_set]

		#dyads=np.asarray(dyads_data_set)
		unique_dyads = np.unique(self.dyads)
		self.dyads_index=[np.where(self.dyads == dyad) for dyad in unique_dyads]


	def vectorization(self, tfidf=0, most_predict=0):
		self.tokens, index_data_set = remove_empty(self.tokens)
		self.set_index(index_data_set)
		vectors = []
		for i in range(len(self.dyads_index)):
			index=self.dyads_index[i]
			token = self.tokens[index[0][0]:index[0][-1]+1]
			X=None
			y=None
			if tfidf or most_predict:
				X=self.utterances[index]
			if most_predict:
				y=self.labels[index]
			vectors = vectors + vectorization(token, tfidf, X, y, most_predict)


		self.vect_data_set=vectors


