import sys

import pandas as pd
import numpy as np

from preprocessing import *
from vectorization_spaCy import *
from data_utils import *

from model_tools import *
from classification import *


from spellchecker import SpellChecker


""" Step 1: Extract data from file """

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
duration=np.delete(np.array(df[6]),0).astype(np.float)


utterances = spell_correction(utterances,1)
#print(type(utterances[0]))






















