"""--------------------------------------------------------------
	Collaborative Acts: 
		Visualize data
	
	python3 visualize_data.py dyad [start_time] [end_time]
-----------------------------------------------------------------"""
import sys

import pandas as pd
import numpy as np

from data_utils import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
NLP: class in .csv file
	0. Dyads
	1. Participant
	2. Id
	3. EAT	
	4. StartTime 
	5. EndTime
	6. Duration
	7. Utterances
	8. Subcategories	
	9. Categories
"""

def dialog(X,y, index, start=0, end=20, type_graph='dialog'):
	X = X[np.arange(start, end)]
	y = y[np.arange(start, end)]	
	
	plt.plot(X,y,'m*')
	for i in range(X.shape[0]): 
		plt.axvline(x=X[i], linewidth=0.25, color='c', linestyle='-.')

	if type_graph=='dialog':
		plt.xlabel('Time (ms)')
		plt.ylabel('Collaborativ Acts')
		plt.title('Types of utterances for dyad: '+index+' ( from '+ str(start)+' to '+str(end)+')')

	if type_graph=='participants':
		plt.xlabel('Time (ms)')
		plt.ylabel('Participant')
		plt.title('Participation for dyad: '+index+' ( from '+ str(start)+' to '+str(end)+')')
	plt.show()
	#plt.savefig()


def histogram(x, index):
	#x = [value1, value2, value3,....]
	y, x, _ = plt.hist(x, density=True, color = 'magenta',
            edgecolor = 'black')# bins = 8, edgecolor='black')

	#print(y*100)

	plt.xlabel('Collaborative Acts')
	plt.title('Repartition of collaborative Acts for dyad '+index)
	plt.show()
	#plt.savefig(index+'_0')
	return y,x



def time(x, y , labels, index):
	mean_time=np.zeros(labels.shape[0])
	min_time=np.zeros(labels.shape[0])
	max_time=np.zeros(labels.shape[0])
	time_by_class=np.zeros(labels.shape[0])
	for label in range(labels.shape[0]):
		ind=np.where(y == labels[label])
		time= x[ind]  
		if time != []:
			time_by_class[label] = sum(time)*100 / sum(x)
			print(labels[label],':')
			print('Time min (%): ', min(time))
			print('Time max (%): ', max(time))
			print('Time mean (%): ', np.mean(time))
			mean_time[label]=np.mean(time/ sum(x))*100 #/ sum(x)
			min_time[label]=min(time) * 100/ sum(x)
			max_time[label]=max(time)* 100/ sum(x)


	plt.plot(labels,mean_time,'m*', labels,min_time,'y*', labels,max_time,'g*')
	for i in range(labels.shape[0]): 
		plt.axvline(x=labels[i],linewidth=0.5, color='c', linestyle='-.')
	plt.xlabel('Collaborative Acts')
	plt.ylabel('Mean time of utterances (%)')
	plt.title('Mean time of utterances according to their category for dyad: '+index)
	plt.show()
	#plt.savefig(index+'_1')

	plt.plot(labels,time_by_class,'m*')
	#print(time_by_class)
	for i in range(labels.shape[0]): 
		plt.axvline(x=labels[i],linewidth=0.5, color='c', linestyle='-.')
	plt.xlabel('Collaborative Acts')
	plt.ylabel('Total time of utterances (%)')
	plt.title('Total time of utterances according to their category for dyad: '+index)
	plt.show()
	#plt.savefig(index+'_2')
			

""" ------------------------------------------ 
	Visualization and study of data 
------------------------------------------ --- """

dyad_ind = int(sys.argv[1]) # from 0 to 18
start_time=0
end_time=20

if len(sys.argv) == 4:
	start_time = int(sys.argv[2])
	end_time = int(sys.argv[3])

""" *** Extract data from file *** """

dataFile='./collaborativeActs.csv'

df = pd.read_csv(dataFile,delimiter="\t",header=None,error_bad_lines=False, encoding="utf8")

# Information about dyads
dyads=np.delete(np.array(df[0]),0)
unique_dyads = np.unique(dyads)
print('Dyads: ', unique_dyads)

# Informations about utterances
utterances=np.delete(np.array(df[7]),0)  
subcategories=np.delete(np.array(df[8]),0)    
categories=np.delete(np.array(df[9]),0) 

collab_acts = np.unique(categories)
print('Categories: ', collab_acts)

# Participants
participants=np.delete(np.array(df[1]),0)

# Time scale
start=np.delete(np.array(df[4]),0)
duration=np.delete(np.array(df[6]),0)

""" *** Visualization *** """
 
print('\n')
#for dyad_ind in range(unique_dyads.shape[0]):
index = np.where(dyads == unique_dyads[dyad_ind])
utterance_dyad = utterances[index]
categories_dyad = categories[index]
labels_dyad = prep_labels(categories_dyad, utterance_dyad, collab_acts)

participants_dyad = participants[index]

start_dyad = start[index].astype(np.float)
duration_dyad = duration[index].astype(np.float)

minut, sec = divmod(int(sum(duration_dyad)),60)
millsec =  sum(duration_dyad)-int(sum(duration_dyad))

print('\nDyad: ',unique_dyads[dyad_ind],' Utterances: ', utterance_dyad.shape, ' Categories: ', categories_dyad.shape) 
print('Time:', sum(duration_dyad), minut,':',sec,':',millsec)
histogram(categories_dyad,unique_dyads[dyad_ind])

dialog(start_dyad,participants_dyad, unique_dyads[dyad_ind], start_time, end_time, 'paticipants')

dialog(start_dyad,categories_dyad, unique_dyads[dyad_ind], start_time, end_time, 'dialog')
time(duration_dyad, categories_dyad, collab_acts, unique_dyads[dyad_ind])


	




