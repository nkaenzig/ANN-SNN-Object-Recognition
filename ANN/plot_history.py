"""
EVENT-BASED OBJECT RECOGNITION USING ANALOG AND SPIKING NEURAL NETWORKS
Semesterproject

plot_history.py
Script used to plot the recorded test accuracies and losses after training stored in .json file
Curve smoothing with 2nd order moving average filter

@author: Nicolas Kaenzig, D-ITET, ETH Zurich
"""

import json
import matplotlib.pyplot as plt
import numpy as np

EPOCHS = 30

def holt_winters_second_order_ewma(x, span, beta, skip=1):
    N = x.size
    alpha = 2.0 / (1 + span)
    s = np.zeros((N,))
    b = np.zeros((N,))
    s[:skip] = x[0:skip]
    for i in range(skip, N):
        s[i] = alpha * x[i] + (1 - alpha) * (s[i - 1] + b[i - 1])
        b[i] = beta * (s[i] - s[i - 1]) + (1 - beta) * b[i - 1]
    return s


histories = []
with open('./model/histories/dvs36_evtaccCOR_maxpool_30E/history.json') as f:
    histories.append(json.load(f))
with open('./model/histories/dvs36_B32_60E_exp/history.json') as f:
    histories.append(json.load(f))


f, axarr = plt.subplots(2, sharex=True)
acc_legend = ['frametype 1A', 'frametype 1B (predator)' ]
# acc_legend = ['64', '32', '16', '8']
# loss_legend = ['APS', 'DVS', 'FULL']
loss_legend = []

for i, history in enumerate(histories):
    axarr[0].set_title('Accuracy')
    #axarr[0].plot(history['acc'][:30])
    #axarr[0].plot(history['val_acc'][:EPOCHS])
    axarr[0].plot(holt_winters_second_order_ewma(np.array(history['val_acc'][:EPOCHS]), 10, 0.1, 5))
    #acc_legend += ['train {}'.format(i), 'test {}'.format(i)]

    axarr[1].set_title('Loss')
    #axarr[1].plot(history['loss'][:30])
    #axarr[1].plot(history['val_loss'][:EPOCHS])
    axarr[1].plot(holt_winters_second_order_ewma(np.array(history['val_loss'][:EPOCHS]), 10, 0.1, 5))
    #loss_legend += ['train {}'.format(i), 'test {}'.format(i)]

axarr[0].legend(acc_legend, loc='lower right')
#axarr[1].legend(loss_legend, loc='upper left')

plt.savefig('./fig/test_frametype.png', format='png')
plt.savefig('./fig/test_frametype.eps', format='eps')

plt.show()