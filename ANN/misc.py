"""
EVENT-BASED OBJECT RECOGNITION USING ANALOG AND SPIKING NEURAL NETWORKS
Semesterproject

misc.py
Contains functions used for data preprocesing and training

@author: Nicolas Kaenzig, D-ITET, ETH Zurich
"""

import numpy as np
import h5py

def three_sigma_frame_clipping_evtsum(frame):
    """
    Clipps all pixel intensities of a frame bigger than 3 times the standard deviation

    :param frame: image frame to be processed
    :return:  3-sigmal clipped image frame
    """
    # Compute standard deviation of event-sum distribution
    # after removing zeros
    sigma = np.std(frame[np.nonzero(frame)])

    # Clip number of events per pixel to three-sigma
    frame = np.clip(frame, 0, 3*sigma)
    return frame



def aps_frame_scaling(frame):
    """
    # Scales frame pixel values to [0,1] to obtain greyscale images

    :param frame: image frame to be processed
    :return: greyscale image
    """
    min_pixel = np.min(frame)
    max_pixel = np.max(frame)

    frame = (frame - min_pixel) / float((max_pixel - min_pixel))
    return frame


def three_sigma_frame_clipping(frame):
    """
    3-sigma clipping used for predator frames. Clipp all pixels with intensies smaller or bigger than 1.5 times std-dev

    :param frame: image frame to be processed
    :return: 3-sigmal clipped image frame
    """
    sigma = np.std(frame)
    mean = np.mean(frame)

    # Clip number of events per pixel to three-sigma, but leave 0.5 entries unchanged as they correspond to zero event counts
    c_l = mean - 1.5*sigma
    c_r = mean + 1.5*sigma
    frame[np.nonzero(frame!=0.5)] = np.clip(frame[np.nonzero(frame!=0.5)], c_l, c_r)
    #frame = np.clip(frame, c_l, c_r)
    return frame


def dvs_frame_scaling(frame):
    """
    Scale frame pixel values to [0,1], while 0.5 remain values unchanged

    :param frame: 
    :return: greyscale image
    """
    min_pixel = np.min(frame)
    max_pixel = np.max(frame)

    frame[np.nonzero(frame != 0.5)] = (frame[np.nonzero(frame != 0.5)] - min_pixel) / float((max_pixel - min_pixel))
    return frame


def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.

    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y)
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.int32)
    categorical[np.arange(n), y-1] = 1 # -1 because labels start at 1 not at 0
    return categorical

def generate_batches_from_hdf5_file(hdf5_file, batch_size, dimensions, num_classes, shuffle):
    """
    Generator that returns batches of images ('xs') and labels ('ys') from a h5 file.
    
    :param hdf5_file: hdf5_file descriptor containing labeled dataset
    :param batch_size: number of samples per batch
    :param dimensions: dimensions of the frames
    :param num_classes: number of output classes
    :param shuffle: if True dataset is shuffled after each epoch
    """
    data_size = len(hdf5_file['labels'])
    indices = np.arange(data_size)

    while 1:
        if shuffle:
            indices = np.random.permutation(indices)

        # count how many entries we have read
        n_entries = 0
        # as long as we haven't read all entries from the file: keep reading
        while n_entries < (data_size - batch_size):
            # start the next batch at index 0
            # create numpy arrays of input data (features)
            if shuffle:
                # indices have to be in increasing order for hdf5 access (unlike numpy...)
                batch_indices = sorted(indices[n_entries: n_entries + batch_size])
            else:
                batch_indices = list(indices[n_entries: n_entries + batch_size])

            xs = hdf5_file['images'][batch_indices]
            xs = np.reshape(xs, dimensions).astype('float32')

            y_values = hdf5_file['labels'][batch_indices]
            #ys = keras.utils.to_categorical(y_values, num_classes)
            ys = to_categorical(y_values, num_classes)

            # we have read one more batch from this file
            n_entries += batch_size
            yield (xs, ys)

        # hdf5_file.close()