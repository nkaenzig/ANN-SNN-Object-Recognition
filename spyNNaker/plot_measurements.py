"""
EVENT-BASED OBJECT RECOGNITION USING ANALOG AND SPIKING NEURAL NETWORKS
Semesterproject

plot_measurements.py
Script used to plot the measured spiketrains and membrane potentials

@author: Nicolas Kaenzig, D-ITET, ETH Zurich
"""

import numpy as np
import matplotlib.pyplot as plt
import neo
import os



spikes = np.load('inputspikes.npz')
spikes = spikes['arr_0']

plt.figure(0)
for (neuron, spike_times) in enumerate(spikes):
    if spike_times == []:
        continue
    #spiking_neurons = spikes.nonzero()
    neuron_vec = np.ones_like(spike_times) * neuron
    plt.plot(spike_times, neuron_vec, '.')
plt.xlabel('Timestep')
plt.ylabel('Neuron')
plt.savefig(os.path.join('./Results', 'input_spikes'), bbox_inches='tight')

pot1 = np.load('pot1.npz')
pot1 = pot1['arr_0']

plt.figure(1)

time_vec = range(pot1.shape[0])
for neuron in range(pot1.shape[1]):
    vmem = [pot1[t][neuron] for t in range(pot1.shape[0])]
    plt.plot(time_vec, vmem)
plt.plot(time_vec, np.ones_like(time_vec) * 0.05, 'r--', label='V_thresh')
plt.plot(time_vec, np.ones_like(time_vec) * 0, 'b-.', label='V_reset')
plt.xlabel('Timestep')
plt.ylabel('Membrane potential')
# for (timestep, vmem) in enumerate(pot1):
#     time_vec = np.ones_like(vmem) * timestep
#     [pot1[i][] for i in range(len(time_vec))]
#     plt.plot(time_vec, vmem)

plt.savefig(os.path.join('./Results', 'pot1'), bbox_inches='tight')
plt.show()