import sys
import pickle

import pandas as pd
import numpy as np

from preprocessing import *
from data_utils import *
from vectorization_spaCy import *



def custom_stopwords(utterance):
	custom_list = ["'fin", "s'cours","'scuse","'traument","'ttend","'être", 'alors',"c'est",'de','donc','la','le', 'les','une', 'un','voilà', 'ah','bah',"c'estse","c'ets","c'qu'on","c'qui","c'que","c't'idée","c'tait","c'te", 'ce','ceci','cela','celle', 'celui','ces','cet','cette','ci',"d'acc","d'accus","d'ins","d'jà", "d'la","d'le","d'leur","d'nos",'de','des','du','en','que','quel','quels', 'quelles', 'quelle', 'zyva']
	custom_list.append(['eh','euh','hein','hm','oh', 'tac'])#'ouf','oups','pff' ])
	tokens = utterance.split(" ")
	tokens= [token for token in tokens if not token in custom_list]
	return (" ").join(tokens)



""" Step 1: Extract data from file """

dataFile='./collaborativeActs.csv'

df = pd.read_csv(dataFile,delimiter="\t",header=None,error_bad_lines=False, encoding="utf8")

utterances = np.delete(np.array(df[7]),0)


tokens_stop, lemma=preprocessing(utterances, 1,1, 1)

stop_words=[]
for i in range(utterances.shape[0]):
	utterance=utterances[i]
	tokens = utterance.split(" ")
	tokens_stop=from_token_to_text(tokens_stop[i])
	diff = list(set(tokens)-set(tokens_stop))

	for d in diff:
		if not d in stop_words:
			stop_words.append(d)

stop_words=np.unique(np.array(stop_words))


print('STOPWORDS')
for i in range(stop_words.shape[0]):
	stop=stop_words[i]
	print(stop)

