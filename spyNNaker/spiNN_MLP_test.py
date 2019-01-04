"""
EVENT-BASED OBJECT RECOGNITION USING ANALOG AND SPIKING NEURAL NETWORKS
Semesterproject

spiNN_MLP_test.py
This module holds a pyNN implementation of a multilayer perceptron (MLP) spiking neurol network written usinng the
spinnaker8 front-end interface to map the network to spiNNaker.

@author: Nicolas Kaenzig, D-ITET, ETH Zurich
"""


import pyNN.spiNNaker as sim
import pyNN.utility.plotting as plot
import matplotlib.pyplot as plt
import misc
import time
import numpy as np
import os



SIMTIME = 250               # simulation timesteps per sample
BATCH_SIZE = 30             # number of samples that are fed into spiNNaker at once
EVENTFRAME_WIDTH = None     # combine EVENTFRAME_WIDTH events with consecutive timesteps into the same simultation step
NO_GAP = True               # remove gaps/phases with no events in the DVS samples
MEASUREMENTS = True         # measure spiketrains and membrane potentials
INHIBITORY = True           # use the MLP model with inhibitory synapses (negaitve weights)
output_spikes = []

if INHIBITORY:
    path = './model/dvs36_evtaccCOR_D16_B0_FLAT_30E/'
else:
    path = './model/dvs36_evtacc_D16_B0_FLAT_posW_10E/'
p1 = path + '01Dense_16'
p2 = path + '02Dense_4'


filepaths, labels = misc.get_sample_filepaths_and_labels('./data/aedat/balanced_100/')

sim.setup(timestep=1.0)

input_pop = sim.Population(size=1296, cellclass=sim.SpikeSourceArray(spike_times=[]), label="spikes")
# to measure input spiketrains introduce an additional population
if MEASUREMENTS:
    pop_0 = sim.Population(size=1296, cellclass=sim.IF_curr_exp(), label="1_pre_input")
    pop_0.set(v_thresh=0.1)
    input_proj = sim.Projection(input_pop, pop_0, sim.OneToOneConnector(), synapse_type=sim.StaticSynapse(weight=3, delay=1))
pop_1 = sim.Population(size=16, cellclass=sim.IF_curr_exp(), label="1_input")
if INHIBITORY:
    pop_1.set(v_thresh=0.05)
else:
    pop_1.set(v_thresh=0.1)
pop_2 = sim.Population(size=4, cellclass=sim.IF_curr_exp(), label="2_hidden")
pop_2.set(v_thresh=0.1)


if INHIBITORY:
    inhibitory_connections_1, exitatory_connections_1 = misc.read_connections(p1)
    inhibitory_connector_1 = sim.FromListConnector(inhibitory_connections_1, column_names=["i", "j", "delay", "weight"])
    exitatory_connector_1 = sim.FromListConnector(exitatory_connections_1, column_names=["i", "j", "delay", "weight"])
    inhibitory_proj_1 = sim.Projection(input_pop, pop_1, inhibitory_connector_1, receptor_type='inhibitory')
    exitatory_proj_1 = sim.Projection(input_pop, pop_1, exitatory_connector_1, receptor_type='excitatory')

    inhibitory_connections_2, exitatory_connections_2 = misc.read_connections(p2)
    inhibitory_connector_2 = sim.FromListConnector(inhibitory_connections_2, column_names=["i", "j", "delay", "weight"])
    exitatory_connector_2 = sim.FromListConnector(exitatory_connections_2, column_names=["i", "j", "delay", "weight"])
    inhibitory_proj_2 = sim.Projection(pop_1, pop_2, inhibitory_connector_2, receptor_type='inhibitory')
    exitatory_proj_2 = sim.Projection(pop_1, pop_2, exitatory_connector_2, receptor_type='excitatory')

else:
    _, connections_1 =  misc.read_connections(p1)
    connector_1 = sim.FromListConnector(connections_1, column_names=["i", "j", "delay", "weight"])
    proj_1 = sim.Projection(input_pop, pop_1, connector_1)

    _, connections_2 = misc.read_connections(p2)
    connector_2 = sim.FromListConnector(connections_2, column_names=["i", "j", "delay", "weight"])
    proj_2 = sim.Projection(pop_1, pop_2, connector_2)

if MEASUREMENTS:
    #pop_0.record(["spikes", "v"])
    pop_1.record(["spikes", "v"])
    pop_2.record(["spikes", "v"])
    #pops = [pop_0, pop_1, pop_2]
    pops = [pop_1, pop_2]
else:
    pop_2.record(["spikes"])
    pops = [pop_2]


# misc.run_testset(sim, SIMTIME, filepaths, labels, input_pop, pop_2, True)
misc.run_testset_sequence(sim, SIMTIME, filepaths, labels, input_pop, pop_2, pops, NO_GAP, 100, 10)
# misc.run_testset_sequence_in_batches(sim, SIMTIME, filepaths, labels, BATCH_SIZE, input_pop, pop_2, pops, NO_GAP, 100, EVENTFRAME_WIDTH)