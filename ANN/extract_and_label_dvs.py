"""
EVENT-BASED OBJECT RECOGNITION USING ANALOG AND SPIKING NEURAL NETWORKS
Semesterproject

extract_and_label_dvs.py
This script was used to create DVS frames from the predator/prey .aedat recordings and to label them.
The processed data is stored in .h5 files

@author: Nicolas Kaenzig, D-ITET, ETH Zurich
"""

from PyAedatTools.ImportAedat import ImportAedat
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm
import h5py
import misc
from tqdm import tqdm
import filenames
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("-rec", "--recording", type=int, help="nunber of the recording to process")
parser.add_argument("-tag", "--tag", type=str, help="tag for hdf5 file name")
args = parser.parse_args()

print('PROCESSING RECORDING NR. ' + str(args.recording))

EULER = False           # set True to run script on EULER computer
RESIZE = False          # set True to resize/subsample the frames to TARGET_DIM
SINGLE_MODE = False     # set True to keep only one event of a patch during subsampling
SCALE_AND_CLIP = True   # 3-sigma clipping and scaling to [0,1]
DVS_FRAME_TYPE = 0      # 0 : integrate pos/negative events for frame accumulation
                        # 1 : use eventcounts/rectified frames
EVENTS_PER_FRAME = 5000
FRAME_DIM = (240,180)
SAVE_FRAME_PLOTS = True
if RESIZE:
    TARGET_DIM = (36,36)
else:
    TARGET_DIM = FRAME_DIM

dim_scale = [FRAME_DIM[0]/float(TARGET_DIM[0]), FRAME_DIM[1]/float(TARGET_DIM[1])]

if EULER:
    inputfile = open('../../../scratch/kaenzign/dvs_targets/' + filenames.target_names[args.recording-1])
else:
    inputfile = open('./data/dvs_targets/' + filenames.target_names[args.recording - 1])

lines = inputfile.readlines()


timestamps = []
labels = []

# labels start at 1 not at 0 --> can't use standard keras to_categorigal fct late
for line in lines:
    line = line.strip()
    if line[0] == '#':
        continue
    timestamps.append(int(line.split(' ')[1]))
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


# Create a dict with which to pass in the input parameters.
aedat = {}
aedat['importParams'] = {}
aedat['info'] = {}

aedat['importParams']['endEvent'] = 3e5;

if EULER:
    aedat['importParams']['filePath'] = '../../../scratch/kaenzign/aedat/' + filenames.aedat_names[args.recording-1]
else:
    aedat['importParams']['filePath'] = './data/aedat/' + filenames.aedat_names[args.recording - 1]

aedat = ImportAedat(aedat)


if DVS_FRAME_TYPE == 0:
    img = np.full(TARGET_DIM, 0.5)
if DVS_FRAME_TYPE == 1:
    img = np.zeros(TARGET_DIM)


i = 0
k = 0
last_j = 0
filenames = [] #for gif generation
tmp_frame_timestamps = []
NR_FRAMES = int(len(aedat['data']['polarity']['timeStamp'])/EVENTS_PER_FRAME)
frame_labels = np.zeros(NR_FRAMES+1)


if EULER:
    hdf5_name = '../../../scratch/kaenzign/processed/dvs_recording' + str(args.recording)
else:
    hdf5_name = './data/processed/dvs_recording' + str(args.recording)
if RESIZE:
    hdf5_name += '_' + str(TARGET_DIM[0]) + 'x' + str(TARGET_DIM[1])
# hdf5_name += '_' + str(int(time.time()))
if args.tag:
    hdf5_name += '_' + args.tag
hdf5_name += '.hdf5'


f = h5py.File(hdf5_name, "w")
d_img = f.create_dataset("images", (NR_FRAMES,TARGET_DIM[0],TARGET_DIM[1]), dtype='f')
d_label = f.create_dataset("labels", (NR_FRAMES,), dtype='i')

last_x = aedat['data']['polarity']['x'][0]
last_y = aedat['data']['polarity']['y'][0]
last_t = aedat['data']['polarity']['timeStamp'][0]


first_frame_event = True
omitted = 0

for t,x,y,p in tqdm(zip(aedat['data']['polarity']['timeStamp'], aedat['data']['polarity']['x'], aedat['data']['polarity']['y'], aedat['data']['polarity']['polarity'])):
    if RESIZE:
        x = int(x/dim_scale[0])
        y = int(y/dim_scale[1])

    if DVS_FRAME_TYPE == 0:
        if p==True:
            img[TARGET_DIM[0]-1-x][TARGET_DIM[1]-1-y] += 0.005
        else:
            img[TARGET_DIM[0]-1-x][TARGET_DIM[1]-1-y] -= 0.005

    if DVS_FRAME_TYPE == 1:
        if SINGLE_MODE:
            if t != last_t or x != last_x or y != last_y or first_frame_event:
                img[TARGET_DIM[0] - 1 - x][TARGET_DIM[1] - 1 - y] += 1
            else:
                #print('time_cluster ommited at frame {}'.format(i))
                omitted += 1

        else:
            img[TARGET_DIM[0]-1-x][TARGET_DIM[1]-1-y] += 1

    tmp_frame_timestamps.append(t)

    last_x = x
    last_y = y
    last_t = t
    first_frame_event = False

    i += 1

    if i%EVENTS_PER_FRAME == 0:
        if SCALE_AND_CLIP:
            if DVS_FRAME_TYPE == 0:
                img = misc.three_sigma_frame_clipping(img)
                img = misc.dvs_frame_scaling(img)

            if DVS_FRAME_TYPE == 1:
                img = misc.three_sigma_frame_clipping_evtsum(img)
                img = misc.aps_frame_scaling(img)

        if SAVE_FRAME_PLOTS:
            fig = plt.imshow((img).T, cmap='gray', norm=NoNorm(vmin=0, vmax=1, clip=True))
            plt.axis('off')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            plt.savefig('./fig/{}_{}_{}.png'.format(k, DVS_FRAME_TYPE, TARGET_DIM), bbox_inches='tight', pad_inches = 0)
            plt.savefig('./fig/{}_{}_{}.eps'.format(k, DVS_FRAME_TYPE, TARGET_DIM), format='eps', bbox_inches='tight', pad_inches = 0) # dpi=1000 ??
            filenames.append('./fig' + "noisy_metro_" + str(k) + ".png")

        for j in range(last_j,len(timestamps)):
            if timestamps[j] > tmp_frame_timestamps[-1]:
                if k>0:
                    d_label[k] = d_label[k-1]
                break

            # take the label of the first timestamp that matches
            if timestamps[j] in tmp_frame_timestamps:
                d_label[k] = labels[j]
                last_j = j
                # print(j,k,labels[j])
                break

        tmp_frame_timestamps = []
        d_img[k] = img

        if DVS_FRAME_TYPE == 0:
            img = np.full(TARGET_DIM, 0.5)
        if DVS_FRAME_TYPE == 1:
            img = np.zeros(TARGET_DIM)
        k += 1
        first_frame_event = True


# Fill up the labels of the first frames
i=0
for label in d_label:
    if label != 0:
        d_label[:i] = label
    else:
        i += 1

print('{} events omitted to avoid subsampling timeclusters'.format(omitted))