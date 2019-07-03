# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 20:47:29 2019

@author: Victor Costa
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import ml_slippage
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

sensor_data = scipy.io.loadmat('UniMiB SHAR dataset/two_classes_data.mat')['two_classes_data']
labels_data = scipy.io.loadmat('UniMiB SHAR dataset/two_classes_labels.mat')['two_classes_labels'][:,0]

sensor_data_h = sensor_data[:, 0:151]
sensor_data_j = sensor_data[:, 151:302]
sensor_data_k = sensor_data[:, 302:453]

# Extract signal magnitude vector (SMV)
# sig_mag_vec = np.array([])
# for i in xrange(0, 151):
    # smv_column = sensor_data[:, i]**2 + sensor_data[:, i+151]**2 + sensor_data[:, i+302]**2
    # if i == 0:
        # sig_mag_vec = smv_column
    # else:
        # sig_mag_vec = np.column_stack((sig_mag_vec, smv_column))

## Extract features        
# transformed_windows = []
# #for window in sig_mag_vec:    
# for window  in sensor_data_j:
    # average = sum(window)/len(window)
    # #std = np.std(window, ddof=1)
    # #window = (window - average)/std;
    # window = window - average
    # z_cross_count = 0
    # for i, sample in enumerate(window):
        # if i > 0 and window[i]*window[i-1] < 0:
            # z_cross_count += 1
    # transformed_windows.append([average, z_cross_count])
    
# print np.array(transformed_windows).shape
# print transformed_windows[-200:], labels_data[-200:]


# X_train, X_test, y_train, y_test = train_test_split(transformed_windows, labels_data, shuffle=True, test_size=0.3, random_state=42)
#X_train, X_test, y_train, y_test = train_test_split(sig_mag_vec, labels_data, test_size=0.33, random_state=42)
#clf = SGDClassifier(class_weight='balanced', early_stopping=False,loss='log', penalty='l2', max_iter=2000, tol=1e-5, shuffle=False, n_iter_no_change=10, random_state = 42)
#clf = RandomForestClassifier(random_state=42)
# clf = LinearSVC(random_state=42, C = 0.1, max_iter = 5000)
# clf.fit(X_train, y_train)
# print "Decision vector = " + str(clf.coef_)
# print "Score = " + str(clf.score(X_test, y_test))
#print clf.predict(X_test[0:20])
#print clf.decision_function(X_test[0:20])

## Plotting
flat_h = sensor_data_h.flatten()
flat_j = sensor_data_j.flatten()
flat_k = sensor_data_k.flatten()
# flat_smv = np.sqrt(flat_h**2 + flat_j**2 + flat_k**2)
# flat_labels = []
# for i in xrange(0, len(flat_h)):
    # windowed_index = i/151
    # flat_labels.append(labels_data[windowed_index])

# print len(flat_labels), len(labels_data), len(flat_h)
    
## Plot raw axis
# plt.figure(figsize=(20,1))
# plt.subplot(3,1,1)
# plt.plot(flat_h, color='k', linewidth=1, alpha = 0.8)
# plt.plot(np.repeat(20*(labels_data-1),151), color='g')

# plt.subplot(3,1,2)
# plt.plot(flat_j, color='k', linewidth=1, alpha = 0.8)
# plt.plot(np.repeat(20*(labels_data-1),151), color='g')

# plt.subplot(3,1,3)
# plt.plot(flat_k, color='k', linewidth=1, alpha = 0.8)
# plt.plot(np.repeat(20*(labels_data-1),151), color='g')

# plt.show()

## Plot SMV
# plt.figure(figsize=(20,1))
# plt.plot(flat_smv, color='k', linewidth=1, alpha = 0.8)
# plt.plot(np.repeat(20*(labels_data-1),151), color='g')
# plt.show()

## Moving Average
# mov_avg_h, avg_labels = ml_slippage.apply_average(flat_h, flat_labels) 
# mov_avg_j, _ = ml_slippage.apply_average(flat_j, flat_labels) 
# mov_avg_k, _ = ml_slippage.apply_average(flat_k, flat_labels) 

# ## Plot
# plt.figure(figsize=(20,1))
# plt.subplot(3,1,1)
# plt.plot(mov_avg_h, color='k', linewidth=1, alpha = 0.8)
# plt.plot(20*(np.array(flat_labels)-1), color='g')

# plt.subplot(3,1,2)
# plt.plot(mov_avg_j, color='k', linewidth=1, alpha = 0.8)
# plt.plot(20*(np.array(flat_labels)-1), color='g')

# plt.subplot(3,1,3)
# plt.plot(mov_avg_k, color='k', linewidth=1, alpha = 0.8)
# plt.plot(20*(np.array(flat_labels)-1), color='g')

# plt.show()

## Exponential Average (Leaky Integrator)
# alpha = 0.07
# expo_avg_h, _ = ml_slippage.apply_exponential_average(flat_h, flat_labels, alpha) 
# expo_avg_j, _ = ml_slippage.apply_exponential_average(flat_j, flat_labels, alpha) 
# expo_avg_k, _ = ml_slippage.apply_exponential_average(flat_k, flat_labels, alpha) 

# ## Plot
# plt.figure(figsize=(20,1))
# plt.subplot(3,1,1)
# plt.plot(expo_avg_h, color='k', linewidth=1, alpha = 0.8)
# plt.plot(20*(np.array(flat_labels)-1), color='g')

# plt.subplot(3,1,2)
# plt.plot(expo_avg_j, color='k', linewidth=1, alpha = 0.8)
# plt.plot(20*(np.array(flat_labels)-1), color='g')

# plt.subplot(3,1,3)
# plt.plot(expo_avg_k, color='k', linewidth=1, alpha = 0.8)
# plt.plot(20*(np.array(flat_labels)-1), color='g')

# plt.show()

## Save raw data
# stacked_flat = np.column_stack((np.column_stack((flat_h, flat_j)), flat_k))
# np.savetxt('3d_full_data.csv', stacked_flat, delimiter=',')
stacked_flat_partial = np.vstack((np.column_stack((np.column_stack((flat_h, flat_j)), flat_k))[0:10000,:], np.column_stack((np.column_stack((flat_h, flat_j)), flat_k))[-10000:-1,:]))
np.savetxt('3d_partial_data.csv', stacked_flat_partial, delimiter=',')