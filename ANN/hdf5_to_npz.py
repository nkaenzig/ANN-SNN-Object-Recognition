"""
EVENT-BASED OBJECT RECOGNITION USING ANALOG AND SPIKING NEURAL NETWORKS
Semesterproject

hdf5_to_npz.py
Used to create npz files for ANN-SNN toolbox
x_norm.npz contains a fraction of the trainset to be used for normalizing weights and biases
x_test.npz contains test frames
y_test.npz contains labels of the test frames


@author: Nicolas Kaenzig, D-ITET, ETH Zurich
"""

import numpy as np
import h5py
import misc
from random import shuffle

def balance_classes(x,y):
    counts = [0,0,0,0]

    indexes = {}
    indexes['N'] = []
    indexes['L'] = []
    indexes['C'] = []
    indexes['R'] = []

    for i, label in enumerate(y):
        if label == 1:
            counts[0] += 1
            indexes['N'].append(i)
        elif label == 2:
            counts[1] += 1
            indexes['L'].append(i)
        elif label == 3:
            counts[2] += 1
            indexes['C'].append(i)
        elif label == 4:
            counts[3] += 1
            indexes['R'].append(i)

    min_class_count = min(counts)
    max_class_count = max(counts)
    print('class {} has minimal count {}'.format(counts.index(min_class_count), min_class_count))
    print('class {} has minimal count {}'.format(counts.index(max_class_count), max_class_count))

    idx = indexes['N'][:min_class_count] + indexes['L'][:min_class_count] + indexes['C'][:min_class_count] + indexes['R'][:min_class_count]
    shuffle(idx)

    return x[idx], y[idx]


NR_SAMPLES = 2000
BALANCE_CLASSES = True
TESTFILE = 'test.hdf5'
TRAINFILE = 'train.hdf5'
test_data_path = './data/processed/dvs_36_evtacc/' + TESTFILE
train_data_path = './data/processed/dvs_36_evtacc/' + TRAINFILE

DENSE = True

test_h5 = h5py.File(test_data_path,'r')
train_h5 = h5py.File(train_data_path,'r')

random_indexes = np.random.randint(0, test_h5['images'].shape[0], size=NR_SAMPLES)

x_test = np.array(test_h5['images'])[random_indexes]
if DENSE:
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
else:
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2])

y_test = np.array(test_h5['labels'])[random_indexes]
if BALANCE_CLASSES:
    x_test, y_test = balance_classes(x_test, y_test)

y_test = misc.to_categorical(y_test)

random_indexes = np.random.randint(0, train_h5['images'].shape[0], size=int(0.2*train_h5['images'].shape[0]))
x_norm = np.array(train_h5['images'])[random_indexes]
if DENSE:
    x_norm = x_norm.reshape(x_norm.shape[0], x_norm.shape[1]*x_norm.shape[2])
else:
    x_norm = x_norm.reshape(x_norm.shape[0], x_norm.shape[1], x_norm.shape[2], 1)

np.savez(file='x_test', arr_0=x_test)
np.savez(file='y_test', arr_0=y_test)
np.savez(file='x_norm', arr_0=x_norm)