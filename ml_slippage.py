# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 17:24:30 2019

@author: Victor Costa
"""



import random
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from sklearn import preprocessing
from scipy import signal
from scipy import stats
from sklearn.metrics import confusion_matrix
# from pyentrp import entropy as ent
# import tsfresh
# import pywt

window_size = 64
Fs = 200.0
window_overlap = 0.75

# Preprocessing
def apply_pos_derivative(signal, labels):
    sig_pos_derivative = []
    labels_pos_derivative = []
    for index, sample in enumerate(signal):
        if index >= 2:
            derivative = (signal[index]-signal[index-2])
            if derivative < 0:
                derivative = 0
            sig_pos_derivative.append(derivative)
            labels_pos_derivative.append(labels[index])
            
    return sig_pos_derivative, labels_pos_derivative

def apply_derivative(signal, labels):
    sig_derivative = []
    labels_derivative = []
    for index, sample in enumerate(signal):
        if index >= 2:
            derivative = (signal[index]-signal[index-2])
            sig_derivative.append(derivative)
            labels_derivative.append(labels[index])
            
    return sig_derivative, labels_derivative

def apply_exponential_average(signal, labels, alpha):
    labels_avg = labels[1:]
    signal_avg = []
    past_avg = 0.0
    current_avg = 0.0
    if alpha > 0 and alpha < 1:
        for i in xrange(1,len(signal)):
            current_avg = alpha*signal[i] + (1.0 - alpha)*past_avg
            past_avg = current_avg
            signal_avg.append(current_avg)
    else:
        print "(expo avg) invalid alpha"
        
    return signal_avg, labels_avg
    
def apply_average(signal, labels):
    
    sig_average = []
    labels_average = []

    avg_window_size = 48
    
    for i in xrange(avg_window_size-1, len(signal)):
        samples = signal[i-avg_window_size+1:i+1]
        average = np.mean(samples)
        sig_average.append(average)
        labels_average.append(labels[i])

    return sig_average, labels_average

## MACHINE LEARNING PIPELINE METHODS
def digital_filter(raw_signal):
    """ Given raw_signal (np array), outputs its filtered version """

    Nyq_freq = Fs * 0.5
    freq_lo_hz = 0.1
    freq_hi_hz = 10.0
    freq_lo_norm = freq_lo_hz/Nyq_freq
    freq_hi_norm = freq_hi_hz/Nyq_freq
    
    z, p, k = signal.iirfilter(N=2, Wn = [freq_lo_norm,freq_hi_norm], btype='bandpass', analog=False, ftype='butter', output='zpk')
    sos = signal.zpk2sos(z,p,k)                                             # Zero, pole, gain to cascaded 2nd order
    filtered_signal = signal.sosfilt(sos, raw_signal)                       # Apply filter
    
    return filtered_signal

def parameterized_digital_filter(frequency_list):
    def design_and_apply(raw_signal, index):
        """ Given raw_signal (np array), outputs its filtered version """
        Nyq_freq = Fs * 0.5
        freq_lo_hz = frequency_list[index][0]
        freq_hi_hz = frequency_list[index][1]
        
        freq_lo_norm = freq_lo_hz/Nyq_freq
        freq_hi_norm = freq_hi_hz/Nyq_freq
        z, p, k = signal.iirfilter(N=8, Wn = [freq_lo_norm,freq_hi_norm], btype='bandpass', analog=False, ftype='butter', output='zpk')
        sos = signal.zpk2sos(z,p,k)                                             # Zero, pole, gain to cascaded 2nd order
        filtered_signal = signal.sosfilt(sos, raw_signal)                       # Apply filter
        
        return filtered_signal
    
    return design_and_apply

def extract_features_window(data):
    """ Given a np-like array, extract desired features and return them """    
    
    Ts = 1/Fs
    
    magnitude = data[-1]
    first_derivative = (data[-1] - data[-3])/(2*Ts)
    
    _, minmax, window_mean, window_var, window_skew, window_kurt = stats.describe(data, axis=0, ddof=1, bias=True, nan_policy='raise')
    window_min = minmax[0]
    window_max = minmax[1]
    my_tfeatures = [first_derivative] # magnitude, window_mean, window_var] # , window_skew, window_kurt, window_min, window_max]
    
    #perm_entropy = ent.permutation_entropy(data_slice, order = 3, delay = 1)
    #second_derivative = (data_slice[-1] - 2*data_slice[-2] + data_slice[-3])/(Ts**2)
    
    #ts_autocorr = tsfresh.feature_extraction.feature_calculators.autocorrelation(data,2)
    ts_longest_strike_above = tsfresh.feature_extraction.feature_calculators.longest_strike_above_mean(data)
    ts_longest_strike_below = tsfresh.feature_extraction.feature_calculators.longest_strike_below_mean(data)
    ts_sec_deriv = tsfresh.feature_extraction.feature_calculators.mean_second_derivative_central(data)
    ts_mean_abs_change = tsfresh.feature_extraction.feature_calculators.mean_abs_change(data)
    tsfr_features = [ts_longest_strike_above, ts_longest_strike_below, ts_sec_deriv, ts_mean_abs_change]
    
    # PSD features
    # _, psd = signal.periodogram(data, fs=Fs, window='flattop', scaling='spectrum')
    _, psd = signal.welch(data, return_onesided=True, fs=Fs, nperseg=window_size)
    psd_features =  list(psd)
    
    #cAprox, cDetail = pywt.dwt(data, 'sym2')
    
    feature_vector = my_tfeatures #+ tsfr_features + psd_features
    #feature_vector = data

    return feature_vector
    
def extract_features_full(X_raw, y_raw):
    sampling_step = int((1-window_overlap) * window_size)                   # 75% overlapping
    
    X_extracted = []
    y_extracted = []
    
    for i in xrange(window_size-1, len(X_raw), sampling_step):
        samples = X_raw[i-window_size+1:i+1]
        hann_window = signal.hanning(M=len(samples), sym=False)
        
        samples = samples * hann_window
        features_vec = extract_features_window(samples)
        X_extracted.append(features_vec)
        y_extracted.append(y_raw[i])
    
    return X_extracted, y_extracted

def extract_features_exclusive(X_raw, y_raw):
    ''' extracts features without passing through intersection regions '''
    sampling_step = int((1-window_overlap) * window_size)                   # 75% overlapping
    
    X_final = []
    y_final = []
    X_non_slip = []
    y_non_slip = []
    X_slip = []
    y_slip = []
    
    non_slip_indices = [indx for indx, label in enumerate(y_raw) if label == 0]
    slip_indices = [indx for indx, label in enumerate(y_raw) if label == 1]
    
    # non slip samples
    for i in xrange(window_size-1, len(non_slip_indices), sampling_step):
        samples = X_raw[non_slip_indices[i]-window_size+1:non_slip_indices[i]+1]
        hann_window = signal.hanning(M=len(samples), sym=False)
        samples = samples * hann_window
        features_vec = extract_features_window(samples)
        X_non_slip.append(features_vec)
        y_non_slip.append(y_raw[non_slip_indices[i]])
    
    # slip samples
    for i in xrange(window_size-1, len(slip_indices), sampling_step):
        samples = X_raw[slip_indices[i]-window_size+1:slip_indices[i]+1]
        hann_window = signal.hanning(M=len(samples), sym=False)
        samples = samples * hann_window
        features_vec = extract_features_window(samples)
        X_slip.append(features_vec)
        y_slip.append(y_raw[slip_indices[i]])
    
    X_final = X_slip + X_non_slip
    y_final = y_slip + y_non_slip
    
    return X_final, y_final

def scaler(X_train, X_test):    
    """scaler = preprocessing.StandardScaler().fit(X_train)
    scaled_X_train = scaler.transform(X_train)
    scaled_X_test = scaler.transform(X_test)
    
    return scaled_X_train, scaled_X_test"""
    return X_train, X_test


### CROSS-VALIDATION METHODS
def generate_cv_datasets(experiments_dict,  preprocessor, feature_extractor, postprocessor, train_size=18, validation_size=9):
    """ Given a experiments dictionary, how many experiments will be used to train/validation and the pipeline methods, return train and validation set dictionaries. """
    
    experiments_dict = pkl.load(open('separate_experiments_dict.pkl','rb'))
    all_experiments = ['AA1', 'AB1', 'AC1', 'BA1', 'BB1', 'BC1', 'CA1', 'CB1', 'CC1'] + ['AA2', 'AB2', 'AC2', 'BA2', 'BB2', 'BC2', 'CA2', 'CB2', 'CC2'] + ['AA3', 'AB3', 'AC3', 'BA3', 'BB3', 'BC3', 'CA3', 'CB3', 'CC3']
    train_dict = {}
    validation_dict ={}
    
    for fold in ['1', '2', '3']:
        # Select experiments for train and validation according to their names (fold k uses experiment trials k as validation and the others as train)
        available_train_experiments = [experiment for experiment in all_experiments if fold not in experiment]
        available_validation_experiments = [experiment for experiment in all_experiments if fold in experiment]
        train_experiments = random.sample(available_train_experiments, train_size)
        validation_experiments = random.sample(available_validation_experiments, validation_size)
        
        # Concatenate raw train and validation data
        train_raw = []
        train_labels = []
        for experiment in train_experiments:
            train_raw.extend(experiments_dict[experiment][:,0])
            train_labels.extend(experiments_dict[experiment][:,1])
        validation_raw = []
        validation_labels = []
        for experiment in validation_experiments:
            validation_raw.extend(experiments_dict[experiment][:,0])
            validation_labels.extend(experiments_dict[experiment][:,1])
        
        ## Extract datasets from raw data
        # Preprocessing
        #X_train = preprocessor(train_raw)
        #X_validation = preprocessor(validation_raw)
        X_train = train_raw
        X_validation = validation_raw
        # Feature extraction
        X_train, y_train = feature_extractor(X_train, train_labels)
        X_validation, y_validation = feature_extractor(X_validation, validation_labels)
        # Postprocessing
        X_train = np.array(X_train)
        X_validation = np.array(X_validation)
        X_train, X_validation = postprocessor(X_train, X_validation)
        
        # Store in dictionary
        train_dict[fold] = np.column_stack((X_train, y_train))
        validation_dict[fold] = np.column_stack((X_validation, y_validation))
        
    return train_dict, validation_dict

def classic_cv_error(train_dict, validation_dict, model_scorer):
    """ Implements the cross-validation structure over 3 folds defined in cv_create_folds.py, using fully parameterized pipeline """
    folds = ['1','2','3']
    partial_errors = np.zeros(len(folds))
    
    for fold in folds:
        #print "[Using experiments \"%c\" as validation set]" % fold
        X_train = train_dict[fold][:,0:-1]
        y_train = train_dict[fold][:,-1]
        X_validation = validation_dict[fold][:,0:-1]
        y_validation = validation_dict[fold][:,-1] 
                
        # Train and get score
        #estimator_accuracy = model_scorer(X_train, y_train, X_validation, y_validation)
        estimator_error = model_scorer(X_train, y_train, X_validation, y_validation)
        #print "Estimator accuracy: %f" % estimator_accuracy
        
        # Keep ensemble error on the current fold
        index = ord(fold) - 49                              # Char to index
        #partial_errors[index] = 1.0 - estimator_accuracy
        partial_errors[index] = estimator_error
        
    average_error = np.sum(partial_errors)/len(folds)
    return average_error

def ew_ensemble_cv_error(experiments_dict, validation_dict, preprocessor, feature_extractor, postprocessor, model_list):
    """ FUTURE: USE generate_cv_dataset() TO GENERATE DATASETS OUT OF THE LOOP.
        Implements the cross-validation structure over 3 folds defined in cv_create_folds.py, together with bagging-like experiment-wise ensemble.
        Use different models and different preprocessors """
    all_experiments = ['AA1', 'AB1', 'AC1', 'BA1', 'BB1', 'BC1', 'CA1', 'CB1', 'CC1'] + ['AA2', 'AB2', 'AC2', 'BA2', 'BB2', 'BC2', 'CA2', 'CB2', 'CC2'] + ['AA3', 'AB3', 'AC3', 'BA3', 'BB3', 'BC3', 'CA3', 'CB3', 'CC3']
    folds = ['1','2','3']
    partial_errors = np.zeros(len(folds))
    experiments_used_to_train = 6
    
    for fold in folds:
        print "[Using experiments \"%c\" as validation set]" % fold
        
        # List comprehension to select which experiments will be used as train for each fold.
        train_exps_available = [exp for exp in all_experiments if fold not in exp]
        
        ensemble_size = len(model_list)
        individual_predictions = []
        individual_scores = []
        for member in xrange(ensemble_size):
            # Define validation raw data
            raw_validation = validation_dict[fold][:,0]
            labels_validation = validation_dict[fold][:,1]
            # Define random subset of experiments that will be used to train each model (2/3 of the experiments)
            train_exps_chosen = random.sample(train_exps_available, experiments_used_to_train)
            # Concatenate signals of the chosen experiments
            raw_train = []
            labels_train = []
            for experiment in train_exps_chosen:
                raw_train.extend(experiments_dict[experiment][:,0])
                labels_train.extend(experiments_dict[experiment][:,1]) 
            
            # Apply preprocessing
            X_train = preprocessor(raw_train, member)
            X_validation = preprocessor(raw_validation, member)
            
            figname = "filtered_%d" % member
            plt.figure(figsize=(150,5))
            plt.plot(raw_train, color='k')
            plt.savefig(figname)
            plt.close()
            
            # Extract features
            X_train, y_train = feature_extractor(X_train, labels_train)
            X_validation, y_validation = feature_extractor(X_validation, labels_validation)
            # Apply postprocessing
            X_train, X_validation = postprocessor(X_train, X_validation)
            # Fit model and save its predictions on validation set
            # MOVER O ABAIXO PARA UM MÉTODO SEPARADO, NO MESMO MOLDE DO KERNEL PCA
            # MODIFICAR FOR PARA FAZER MEMBER PERCORRER MODEL_LIST
            model_list[member].fit(X_train, y_train)
            prediction = model_list[member].predict(X_validation)
            individual_score = model_list[member].score(X_validation, y_validation)
            print "Model %d score = %f" % (member, individual_score) 
            individual_scores.append(individual_score)
            individual_predictions.append(prediction)
        print "Ensemble models mean score = %f" % (np.mean(individual_scores))
        # Extract ensemble predictions and accuracy
        ensemble_predictions = (np.sum(individual_predictions, axis=0) > float(ensemble_size)/2) * 1
        hits = np.sum(np.equal(ensemble_predictions, y_validation))
        ensemble_accuracy = float(hits)/float(len(y_validation))
        print "Ensemble accuracy: %f" % ensemble_accuracy
        # Keep ensemble error on the current fold
        partial_errors[ord(fold) - 49] = 1.0 - ensemble_accuracy
        
    average_error = np.sum(partial_errors)/len(folds)
    return average_error



def classic_model_score(model):
    """ Given a model, trains it on train set and return its accuracy on test set """
    def fit_and_score(X_train, y_train, X_test, y_test):
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        return accuracy
    
    return fit_and_score

def classic_bagging_score(model_list):
    """" Given a list of models, trains a bagging meta-estimator on train set, and return the ensemble accuracy on test set"""
    def fit_and_score(X_train, y_train, X_test, y_test):
        X_train = np.array(X_train);    y_train = np.array(y_train)
        ensemble_size = len(model_list)
        individual_predictions = []
        for index in xrange(ensemble_size):
            # Define bootstrap indices, replicate and labels to stabilish train sets for bagging
            bootstrap_indices = np.random.randint(0,len(X_train),len(X_train))
            bootstrap_replicate = X_train[bootstrap_indices]
            bootstrap_labels = y_train[bootstrap_indices]
            # Fit model and save its predictions on validation set
            model_list[index].fit(bootstrap_replicate, bootstrap_labels)
            prediction = model_list[index].predict(X_test)
            #print "Model %d individual score = %f" % (index, model_list[index].score(X_test, y_test)) 
            individual_predictions.append(prediction)
        # Extract ensemble predictions and accuracy
        ensemble_predictions = (np.sum(individual_predictions, axis=0) > float(ensemble_size)/2) * 1
        hits = np.sum(np.equal(ensemble_predictions, y_test))
        ensemble_accuracy = float(hits)/float(len(y_test))
        return ensemble_accuracy
    
    return fit_and_score


### HYSTERESIS METHODS
def hysteresis_predict(th_0, th_1, fitted_model, X_test):
    """ Given adaptative thresholds and a fitted model, return hysteresis predictions on test set """
    hysteresis_predictions = []
    current_class = 0
    
    for instance in X_test:
        predicted_probability = (fitted_model.predict_proba([instance])).flatten()
        
        if (current_class == 0) and (predicted_probability[1] > 0.5 + th_0):
            current_class = 1
        elif (current_class == 1) and (predicted_probability[0] > 0.5 + th_1):
            current_class = 0
        hysteresis_predictions.append(current_class)
    
    return hysteresis_predictions

def hysteresis_error(th_0, th_1, model):
    """ Given adaptative thresholds ( -0.5 < th < 0.5), fits model on train set, and evaluates the adaptative thresholds on test set """
    def fit_and_score(X_train, y_train, X_test, y_test):
        model.fit(X_train, y_train)
        hysteresis_predictions = hysteresis_predict(th_0, th_1, model, X_test)
        
        conf_matrix = confusion_matrix(y_test, hysteresis_predictions)
        TN = conf_matrix[0][0]
        FP = conf_matrix[0][1]
        TP = conf_matrix[1][1]
        FN = conf_matrix[1][0]
    
        sensitivity = float(TP)/float(TP+FN)
        specificity = float(TN)/float(TN+FP)
        ss_mean = (sensitivity+specificity)/2
    
        error = ((1 - ss_mean)**4) * np.sqrt(ent.permutation_entropy(hysteresis_predictions, order=2))
        
        return error
    
    return fit_and_score

def hysteresis_accuracy(th_0, th_1, model):
    """ Given adaptative thresholds ( -0.5 < th < 0.5), fits model on train set, and returns the accuracy on test set """
    def fit_and_score(X_train, y_train, X_test, y_test):
        model.fit(X_train, y_train)
        hysteresis_predictions = hysteresis_predict(th_0, th_1, model, X_test)
        
        conf_matrix = confusion_matrix(y_test, hysteresis_predictions)
        TN = conf_matrix[0][0]
        FP = conf_matrix[0][1]
        TP = conf_matrix[1][1]
        FN = conf_matrix[1][0]
        
        accuracy = float(TP+TN)/float(TP+TN+FP+FN)
        #hits = np.sum(np.equal(hysteresis_predictions, y_test))
        #hysteresis_accuracy = float(hits)/float(len(hysteresis_predictions))
        
        return accuracy
    
    return fit_and_score

