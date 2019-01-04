"""
EVENT-BASED OBJECT RECOGNITION USING ANALOG AND SPIKING NEURAL NETWORKS
Semesterproject

test.py
This module can be used to evaluate accuracies of a stored model on a testset stored in HDF5 fileformat

@author: Nicolas Kaenzig, D-ITET, ETH Zurich
"""


import keras
from keras.models import load_model
import h5py
import misc
import numpy as np
import time

MODEL = 'weights.09-0.38.h5'
model_path = './model/report/dvs36_evtacc_maxpool_B0_10E/' + MODEL

TESTFILE = 'test.hdf5'
test_data_path = './data/processed/dvs_36_evtacc/' + TESTFILE

img_rows = 36
img_cols = 36

DENSE = False # To be used for models with 1-dim inputlayer such as MLPs


test_h5 = h5py.File(test_data_path,'r')

x = test_h5['images'][:]
if DENSE:
    dimensions = (x.shape[0], img_rows*img_cols)
else:
    dimensions = (x.shape[0], img_rows, img_cols, 1)
x = np.reshape(x, dimensions).astype('float32')

y = test_h5['labels'][:]
y = misc.to_categorical(y, 4)


model = load_model(model_path)

start_time = time.time()
predictions = model.predict(x)
delta_t = (time.time() - start_time)/x.shape[0]
print("--- Prediction took %s seconds ---" % delta_t)

predictions = np.argmax(predictions, axis=1)
gnd_truth = np.argmax(y, axis=1)

comparisons = predictions == gnd_truth

correct_class_predictions = [0,0,0,0]
nr_class_samples = [0,0,0,0]

for i, label in enumerate(gnd_truth):
    nr_class_samples[label] += 1
    if comparisons[i] == True:
        correct_class_predictions[label] += 1

correct_predictions = np.sum(comparisons)
tot_accuracy = correct_predictions/float(y.shape[0])

class_accuracies = np.array(correct_class_predictions)/np.array(nr_class_samples,dtype=float)

print('TEST ACCURACY: {}'.format(tot_accuracy))
print('N ACCURACY: {}'.format(class_accuracies[0]))
print('L ACCURACY: {}'.format(class_accuracies[1]))
print('C ACCURACY: {}'.format(class_accuracies[2]))
print('R ACCURACY: {}'.format(class_accuracies[3]))
print('mean accuracy over classes: {}'.format(np.mean(class_accuracies)))

