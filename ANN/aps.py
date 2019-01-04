"""
EVENT-BASED OBJECT RECOGNITION USING ANALOG AND SPIKING NEURAL NETWORKS
Semesterproject

aps.py
This module was used to extract aps frames from DAVIS .avi recordings.
The frames can be supsampled and the number of samples can be increased by over-/underexposing pixel intensities
3-sigma clipping around mean pixel intensities and scaling to [0,1] to optain greyscale images

@author: Nicolas Kaenzig, D-ITET, ETH Zurich
"""

import filenames
import argparse
import numpy as np
import h5py
import misc
import imageio
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm
from scipy.misc import imresize
from skimage.transform import rescale, resize

EULER = True        # set True to run script on EULER computer
RESIZE = True       # set True to resize/subsample the frames to TARGET_DIM
EXPOSE = False      # create 3 samples out of one by under-/overexposing pixel intensities

FRAME_DIM = (240,180)
if RESIZE:
    TARGET_DIM = (36,36)
else:
    TARGET_DIM = FRAME_DIM

parser = argparse.ArgumentParser()
parser.add_argument("-rec", "--recording", type=int, help="nunber of the recording to process")
parser.add_argument("-tag", "--tag", type=str, help="tag for hdf5 file name")
args = parser.parse_args()

if EULER:
    f_targets = open('../../../scratch/kaenzign/dvs_targets/' + filenames.target_names[args.recording-1])
    f_aps_timecodes = open('../../../scratch/kaenzign/aps_timecodes/' + filenames.aps_timecode_names[args.recording-1])
    avi_filename = '../../../scratch/kaenzign/aps_avi/' + filenames.aps_avi_names[args.recording-1]
else:
    f_targets = open('./data/dvs_targets/' + filenames.target_names[args.recording - 1])
    f_aps_timecodes = open('./data/aps_timecodes/' + filenames.aps_timecode_names[args.recording-1])
    avi_filename = './data/aps_avi/' + filenames.aps_avi_names[args.recording-1]


target_lines = f_targets.readlines()


target_timestamps = []
labels = []

for line in target_lines:
    line = line.strip()
    if line[0] == '#':
        continue
    target_timestamps.append(int(line.split(' ')[1]))
    x_coord = int(line.split(' ')[2])

    if x_coord == -1:
        labels.append(1) # N = 1
    elif x_coord < 80:
        labels.append(2) # L = 2
    elif x_coord < 160:
        labels.append(3) # C = 3
    elif x_coord < 240:
        labels.append(4) # R = 4
    else:
        labels.append(5) # Invalid = 5

aps_lines = f_aps_timecodes.readlines()

aps_timecodes = []

for line in aps_lines:
    line = line.strip()
    if line[0] == '#':
        continue
    aps_timecodes.append(int(line.split(' ')[1]))

if EULER:
    hdf5_name = '../../../scratch/kaenzign/processed/aps_recording' + str(args.recording)
else:
    hdf5_name = './data/processed/aps_recording' + str(args.recording)
# hdf5_name += '_' + str(int(time.time()))
if RESIZE:
    hdf5_name += '_' + str(TARGET_DIM[0]) + 'x' + str(TARGET_DIM[1])
if EXPOSE:
    hdf5_name += '_' + 'exp'
if args.tag:
    hdf5_name += '_' + args.tag
hdf5_name += '.hdf5'

f = h5py.File(hdf5_name, "w")
NR_FRAMES = len(aps_timecodes)

if EXPOSE:
    NR_FRAMES = NR_FRAMES*3
d_img = f.create_dataset("images", (NR_FRAMES,TARGET_DIM[0],TARGET_DIM[1]), dtype='f')
d_label = f.create_dataset("labels", (NR_FRAMES,), dtype='i')

i = 0
k = 0
for t_aps in aps_timecodes:
    while (t_aps > target_timestamps[i]) and (t_aps < target_timestamps[-1]):
        i += 1

    if EXPOSE:
        d_label[k:k+3] = labels[i]
        k += 3
    else:
        d_label[k] = labels[i]
        k += 1
    # print(k, t_aps, target_timestamps[i], aps_labels[k])


vid = imageio.get_reader(avi_filename,  'ffmpeg')

k = 0
for image in vid.iter_data():
    #d_img[k] = image
    if RESIZE:
        # img = imresize(misc.aps_frame_scaling(np.moveaxis(image, 2, 0)[0].T), size=(TARGET_DIM), interp='nearest')
        # img = misc.aps_frame_scaling(img)
        img = resize(np.moveaxis(image, 2, 0)[0].T, TARGET_DIM)  # also scales the images to [0,1]!
    else:
        img = misc.aps_frame_scaling(np.moveaxis(image, 2, 0)[0].T) # RGB image shape (240,180,3) --> (3,240,180) ---> [0] (240,180)
    if EXPOSE:
        img_u = 0.3*img
        img_o = 2*img
        img_o[np.nonzero(img_o > 1)] = 1 # clip values bigger than 1

        d_img[k] = img
        d_img[k+1] = img_u
        d_img[k+2] = img_o

        k += 3
    else:
        d_img[k] = img
        k += 1

    # print(image.mean())

# plt.imshow((img).T, cmap='gray', norm=NoNorm(vmin=0, vmax=1, clip=True))