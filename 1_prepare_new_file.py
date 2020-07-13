import csv
import string

import pandas as pd
import numpy as np

from spellnCorrection import *
from preprocessing import *

from spellchecker import SpellChecker
spell = SpellChecker(language='fr')


""" Step 1: Extract data from file """

dataFile='./collaborativeActs.csv'
df = pd.read_csv(dataFile,delimiter="\t",header=None,error_bad_lines=False, encoding="utf8")

X = np.array(df)

""" Step 2: Spell and Correction """

sentence=[]
WRONG = []
for i in range(X.shape[0]):
	if i !=0:
		utterance=X[i][7]
		tokens=normalization(utterance)
		tokens=tokenization(tokens)
		tokens=[token.text for token in tokens]
		tokens_txt=correct_jWORDS(tokens)
		tokens= [token for token in tokens_txt if (not token in string.punctuation or token=='?' or token=='!') ]

		misspelled = spell.unknown(tokens)
		for w in misspelled:
			if w in RIGHT_CORRECTED:
				tokens_txt[tokens_txt.index(w)]=spell.correction(w)

			if w in LIST_1:
				while w in tokens_txt:
					index = LIST_1.index(w)
					indx_txt = tokens_txt.index(w)
					tokens_txt = tokens_txt[0:tokens_txt.index(w)] + CORRECT_1[index]+ tokens_txt[tokens_txt.index(w)+1:len(tokens_txt)+1]

			"""if w in VOCAB_JEUN_2:
				while w in tokens_txt:
					index = VOCAB_JEUN_2.index(w)
					indx_txt = tokens_txt.index(w)
					tokens_txt = tokens_txt[0:tokens_txt.index(w)] + JEUN_CORRECT_2[index]+ tokens_txt[tokens_txt.index(w)+1:len(tokens_txt)+1]"""


			"""if w in VOCAB_JEUN_1 or w in VOCAB_JEUN_2:
				tokens_txt[tokens_txt.index(w)]='EXPRESSION JEUNE'"""

			"""if w in NAME:
				tokens_txt[tokens_txt.index(w)]='NOM' """

			if (w in PONCT) or (w in DELETE) or (w in INTERJ_del):
				tokens_txt.remove(w)

			if w in UNKNOWN:
				tokens_txt.remove(w)
				#print('\n\n',w, '\n', utterance)
				#pass
		
			if not w in WRONG:
				sentence.append(utterance)
				WRONG.append(w)
		
		X[i][7] = " ".join(tokens_txt)
		#print('\n', utterance)
		#print(X[i][7])


""" Step 3: Create new .csv file """
with open('collabacts_correction.csv', mode='w') as collab_file:
	writer = csv.writer(collab_file, delimiter="\t")#, quotechar='"', quoting=csv.QUOTE_MINIMAL)
	for i in range(X.shape[0]):
		writer.writerow(X[i])
"""
print(len(WRONG))
for i in range(len(WRONG)):
	w = WRONG[i]
	if (not w in NOMS_PROPRES) and (not w in PONCT) and (not w in DELETE) and (not w in VOCAB_JEUN) and (not w in INTERJ) and (not w in UNKNOWN) and (not w in LIST_1) and (not w in RIGHT_CORRECTED):
		print(w, spell.correction(w))"""




































