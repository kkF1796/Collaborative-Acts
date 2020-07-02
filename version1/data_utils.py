import random
import functools as fctl
import numpy as np



# Find category of a given utterance
def find_categ(utterance, utterances, categories):
	"""for i in range(len(utterances)):
		if(utterance == utterances[i]):
			break;"""
	return categories[np.where(utterances == utterance)]


# Convert categories into (int) labels
# categories = ["Interaction management", "Social relation", "Task management", "Information", "Transactivity", "Tool", "Other","Outside activity"]
def prep_labels(categories, utterances,collab_acts):
	labels = np.zeros(len(categories))
	"""for k in range(len(categories)):
		if(categories[k] == 'Interaction management'):
			labels[k]=0
		elif (categories[k] == 'Social relation'):
			labels[k]=1
		elif (categories[k] ==  'Task management'):
			labels[k]=2
		elif (categories[k] == 'Information'):
			labels[k]=3
		elif(categories[k] == 'Transactivity'):
			labels[k]=4
		elif(categories[k] == 'Tool'):
			labels[k]=5
		elif(categories[k] == 'Other'):
			labels[k]=6
		elif(categories[k] == 'Outside activity'):
			labels[k]=7
		else:
			print("Unrecognized category for utterance ",k, categories[k],", ",utterances[k])"""
	for k in range(len(collab_acts)):
		index = np.where(categories == collab_acts[k])
		labels[index]=k
	return labels








