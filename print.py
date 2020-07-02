import pandas as pd
import numpy as np

from data_utils import *


dataFile='./collaborativeActs.csv'

df = pd.read_csv(dataFile,delimiter="\t",header=None,error_bad_lines=False, encoding="utf8")

# Information about dyads
dyads=np.delete(np.array(df[0]),0)
unique_dyads = np.unique(dyads)
print('Dyads: ', unique_dyads)

dyads_index=[np.where(dyads == dyad) for dyad in unique_dyads]

# Informations about utterances
utterances = np.delete(np.array(df[7]),0)
subcategories = np.delete(np.array(df[8]),0)
categories = np.delete(np.array(df[9]),0)

collab_acts = np.unique(categories)
print('Categories: ', collab_acts, '\n')

labels = prep_labels(categories, utterances, collab_acts)

# Participants
participants=np.delete(np.array(df[1]),0)

# Time scale
start=np.delete(np.array(df[4]),0)
duration=np.delete(np.array(df[6]),0)


for dyad_ind in range(len(dyads_index)):
	print('\n\n')

	index = dyads_index[dyad_ind]

	utterance_dyad = utterances[index]
	categories_dyad = categories[index]

	labels_dyad = prep_labels(categories_dyad, utterance_dyad, collab_acts)

	participants_dyad = participants[index]

	start_dyad = start[index].astype(np.float)
	duration_dyad = duration[index].astype(np.float)

	minut, sec = divmod(int(sum(duration_dyad)),60)
	millsec =  sum(duration_dyad)-int(sum(duration_dyad))

	print('\nDyad: ',unique_dyads[dyad_ind],' Utterances: ', utterance_dyad.shape) 
	print('Time:', sum(duration_dyad), minut,':',sec,':',millsec)

	for i in range(utterance_dyad.shape[0]):
		print(i,': ',participants_dyad[i], '(', categories_dyad[i], '): ', utterance_dyad[i])
	



