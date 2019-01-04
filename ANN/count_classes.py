"""
EVENT-BASED OBJECT RECOGNITION USING ANALOG AND SPIKING NEURAL NETWORKS
Semesterproject

count_classes.py
Script used to count classes of samples in a .h5 dataset

@author: Nicolas Kaenzig, D-ITET, ETH Zurich
"""

import h5py
import matplotlib.pyplot as plt

EULER = True # set True to run script on EULER computer

if EULER:
    processed_path = '../../../scratch/kaenzign/processed/'
else:
    processed_path = './data/processed/'

processed_path += 'dvs_36/'

hdf5_train = h5py.File(processed_path + 'train.hdf5','r')
hdf5_test = h5py.File(processed_path + 'test.hdf5','r')

labels_train = list(hdf5_train['labels'])
labels_test = list(hdf5_test['labels'])

counts_train = {}
counts_train['N'] = 0
counts_train['L'] = 0
counts_train['C'] = 0
counts_train['R'] = 0

counts_test = {}
counts_test['N'] = 0
counts_test['L'] = 0
counts_test['C'] = 0
counts_test['R'] = 0


for label in labels_train:
    if label == 1:
        counts_train['N'] += 1
    elif label == 2:
        counts_train['L'] += 1
    elif label == 3:
        counts_train['C'] += 1
    elif label == 4:
        counts_train['R'] += 1

for label in labels_test:
    if label == 1:
        counts_test['N'] += 1
    elif label == 2:
        counts_test['L'] += 1
    elif label == 3:
        counts_test['C'] += 1
    elif label == 4:
        counts_test['R'] += 1


print('N_train ' + str(counts_train['N']))
print('L_train ' + str(counts_train['L']))
print('C_train ' + str(counts_train['C']))
print('R_train ' + str(counts_train['R']))

print('N_test ' + str(counts_test['N']))
print('L_test ' + str(counts_test['L']))
print('C_test ' + str(counts_test['C']))
print('R_test ' + str(counts_test['R']))

print('N ' + str(counts_train['N'] + counts_test['N']))
print('L ' + str(counts_train['L'] + counts_test['L']))
print('C ' + str(counts_train['C'] + counts_test['C']))
print('R ' + str(counts_train['R'] + counts_test['R']))



# x_bar = range(1,5)
# plt.bar(x_bar, counts_train['N'], label='bar1')

# bins = list(range(2,5,1))
# plt.hist(labels, bins, histtype='bar', rwidth= 0.8, label='hist1')
#
# plt.show()




