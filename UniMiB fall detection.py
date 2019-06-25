# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 20:47:29 2019

@author: Victor Costa
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

sensor_data = scipy.io.loadmat('UniMiB SHAR dataset/two_classes_data.mat')['two_classes_data']
labels_data = scipy.io.loadmat('UniMiB SHAR dataset/two_classes_labels.mat')['two_classes_labels'][:,0]

sensor_data_h = sensor_data[:, 0:151]
sensor_data_j = sensor_data[:, 151:302]
sensor_data_k = sensor_data[:, 302:453]
flat_h = sensor_data_h.flatten()
flat_j = sensor_data_j.flatten()
flat_k = sensor_data_k.flatten()

## Plot
"""
plt.figure(figsize=(1000,15))
plt.subplot(3,1,1)
plt.plot(flat_h, color='k', linewidth=1, alpha = 0.8)
plt.plot(np.repeat(20*(labels_data-1),151), color='g')

plt.subplot(3,1,2)
plt.plot(flat_j, color='k', linewidth=1, alpha = 0.8)
plt.plot(np.repeat(20*(labels_data-1),151), color='g')

plt.subplot(3,1,3)
plt.plot(flat_k, color='k', linewidth=1, alpha = 0.8)
plt.plot(np.repeat(20*(labels_data-1),151), color='g')

plt.show()
"""

## Save raw data
stacked_flat = np.column_stack((np.column_stack((flat_h, flat_j)), flat_k))
np.savetxt('3d_full_data', stacked_flat, delimiter=',')