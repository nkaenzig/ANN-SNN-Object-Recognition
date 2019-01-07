Event-based object recognition using analog and spiking neural networks
=========================================

In this project object recognition in the context of a robotics predator/prey navigation scenario using analog and spiking neural networks is performed. A dataset containing event-based data acquired by a dynamic vision sensor (DVS) is used to train a convolutional neural network (CNN) using frames obtained by accumulation of the DVS event streams. After that the network is converted into a spiking CNN representation and accuracy is evaluated using synthetically generated input spiketrains as well as real event-based DVS input. The reasons for the loss in accuracy that occurs in the SNN after conversion were analyzed. While very little loss in accuracy arised when driving the spiking network with synthetic spikes, for DVS input a bigger gap in accuracy was measured. The conversion of the biases as well as the sparse and non-uniform nature of the DVS event streams were identified to be the main reasons for the observed loss after conversion. In a last step a simple multilayer perceptron architecture was implemented on the neuromorphic platform spiNNaker, evaluating performance and the feasibility of using the platform for performing direct SNN training using backpropagation inspired algorithms instead of ANN-SNN conversion.

Contents
--------

```
/ANN: contains the python sourcecode used to process the predator/prey dataset and to train the ANN models
    - aedat_cutter.py: This module was used to slice .aedat recordings to get seperate and labeled DVS samples to be used as testset.
    - aps.py: This module was used to extract aps frames from DAVIS .avi recordings. The frames can be supsampled and the number of samples can be increased over-/underexposing pixel intensities. 3-sigma clipping around mean pixel intensities and scaling to [0,1] to optain greyscale images.
    - count_classes.py: Script used to count classes of samples in a .h5 dataset
    - extract_and_label_dvs.py: This script was used to create DVS frames from the predator/prey .aedat recordings and to label them. The processed data is stored in .h5 files
    - filenames.py: Contains filenames for all 20 recordings and the corresponding labels of the predator/prey dataset
    - hdf_test.py: Script used to display/store (for debbug & analysis) some of the processed frames stored in .h5 files
    - hdf5_to_npz.py: Used to create npz files for ANN-SNN toolbox
    - misc.py: Contains functions used for data preprocesing and training
    - models.py: This module contains keras implementation of the implemented CNN and MLP models and can be used for training. During training the model parameters are stored to .h5 files after each epoch
    - plot_history.py: Script used to plot the recorded test accuracies and losses after training stored in .json file Curve smoothing with 2nd order moving average filter
    - test.py: This module can be used to evaluate accuracies of a stored model on a testset stored in HDF5 fileformat
    - /models/ (contains the trained models)
        dvs36_orig_predator:    CNN model replicated from predator/prey paper trained with DVS frames (predator frametype)
        aps36_orig_predator:    CNN model replicated from predator/prey paper trained with APS frames
        full36_orig_predator:   CNN model replicated from predator/prey paper trained with DVS (predator frametype) & APS frames
        dvs36_evtacc:           CNN model replicated from predator/prey paper trained with DVS (rectified frametype) frames
        dvs36_evtacc_maxpool_L2:    CNN model replicated from predator/prey paper trained with DVS (rectified frametype), L2 bias regularizer, max-pooling   
        dvs36_evtacc_avgpool_L2:    CNN model replicated from predator/prey paper trained with DVS (rectified frametype), L2 bias regularizer, average-pooling  
        dvs36_evtacc_maxpool_B0:    CNN model replicated from predator/prey paper trained with DVS (rectified frametype), zero bias, max-pooling
        dvs36_evtacc_avgpool_B0:    CNN model replicated from predator/prey paper trained with DVS (rectified frametype), zero bias, average-pooling
        dvs36_evtacc_maxpool_B0_singlesub: same as dvs36_evtacc_maxpool_B0 with DVS dataset used where only one event is kept during subsampling of a patch
```
    
```    
/spiNNaker:
    - plot_measurements.py: Script used to plot the measured spiketrains and membrane potentials
    - spiNN_MLP_test.py: This module holds a pyNN implementation of a multilayer perceptron (MLP) spiking neurol network written usinng the spinnaker8 front-end interface to map the network to spiNNaker.
    - misc.py: Contains functions used for data preprocesing and batching
    - /model/
        dvs36_evtacc_D16_B0_FLAT_posW: MLP model with one hidden layer of 16 neurons, positive weight constraint, zero bias
        dvs36_evtacc_D16_B0_FLAT: LP model with one hidden layer of 16 neurons, zero bias
```

```       
/snn_toolbox/
    ANN-SNN conversion toolbox - used for the conversion and simulations
``` 
