"""
EVENT-BASED OBJECT RECOGNITION USING ANALOG AND SPIKING NEURAL NETWORKS
Semesterproject

hdf_test.py
Script used to display/store (for debbug & analysis) some of the processed frames stored in .h5 files

@author: Nicolas Kaenzig, D-ITET, ETH Zurich
"""

import matplotlib.pyplot as plt
import numpy as np
import h5py
from matplotlib.colors import NoNorm

RECORDING_NR = 1

hdf5_path = './data/processed/aps_recording1_36x36_exp.hdf5'

hdf5_f = h5py.File(hdf5_path,'r')

frames = hdf5_f['images'][:30]
labels = hdf5_f['labels'][:30]

for k, frame in enumerate(frames):
    plt.imshow((frame).T, cmap='gray', norm=NoNorm(vmin=0, vmax=1, clip=True))
    plt.savefig('./fig/' + "test_" + str(k) + ".png")