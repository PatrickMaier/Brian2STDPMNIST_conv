### Rewrite of Diehl&Cook_MNIST_evaluation.py
###
### Simplified code; produces tables on stdout, no graphics
###
### python MNIST_eval_runs.py NETWORK [labeling=L.npy] [predict=P.npy] [RESET]
### where
### * NETWORK is the string '28x28', '14x14', '28x28x2' or '14x14x2'
### * L.npy is a spikecount of MNIST_run_model (SV'd learning, labeling neurons)
### * P.npy is a spikecount of MNIST_run_model (prediction, based on labeling)
### * list of neuron indices RESET indicates which neurons were reset to random

# Parse and import mandatory NETWORK parameter
import network_param
if __name__ == '__main__':
    network_param.parse()
from network_param import NETWORK

import json
import pickle
import sys
import numpy as np

print('BEGIN')
np.random.seed(0)

##############################################################################
## Constants

data_path = './'                                      # path to all files & dirs
result_path = data_path + 'activity/' + NETWORK + '/' # path to spikecount files

labeling_file = 'labeling.npy'  # default file of spike counts for labeling neurons

predict_file = 'predict.npy'    # default file of spike counts for prediting digits

random_neurons = []  # list of indices of neurons w/ weights reset to random

#show_labeling = False    # print labeling for each neuron if True
show_labeling = True     # print labeling for each neuron if True

show_prediction = False  # print prediction for each image if True
#show_prediction = True   # print prediction for each image if True

show_confusion = False   # display confusion matrix if True; req matplotlib
#show_confusion = True    # display confusion matrix if True; req matplotlib


##############################################################################
## Compute response rates (per neuron)

# Returns a 1d-array of response rates per neuron, ie. the fraction of images
# for which the neuron spiked.
def response_rates(spikes):
    n_images, n_e = spikes.shape
    return np.asarray([(spikes[:,i] > 0).sum()/n_images for i in range(n_e)])


##############################################################################
## Normalise spikes (per image)

# Returns two arrays of the same length as spikes.
# * The first is a row-wise normalized array of spike counts (i.e. each row
#   divided by the sum of spikes for that input image).
# * The second is an array of booleans which is False if and only if no spikes
#   were recorded for the image.
def normalize_spikes(spikes):
    norm_exists = np.sum(spikes, axis=1) > 0
    norm_spikes = np.zeros(spikes.shape)
    for i in range(len(spikes)):
        if norm_exists[i]:
            norm_spikes[i] = spikes[i] / np.sum(spikes[i])
    return norm_spikes, norm_exists


#############################################################################
## Compute probability distribution of labels (per neuron)

# Returns an array of probability distributions over the set of labels,
# one distribution per neuron.
def probability_labels(spikes, labels):
    assert(len(spikes) == len(labels) and len(spikes.shape) == 2)
    space = set(labels)
    n_classes = len(space)
    n_e = spikes.shape[1]
    norm_spikes, norm_exists = normalize_spikes(spikes)
    norm_spikes_all = np.sum(norm_spikes, axis=0)
    neuron_active = norm_spikes_all > 0.
    norm_spikes_all[neuron_active == False] = -1.  # hack to prevent div by 0
    norm_spikes_by_label = np.zeros((n_classes, n_e))
    for i,l in enumerate(space):
        norm_spikes_by_label[i] = np.sum(norm_spikes[labels == l], axis=0) / norm_spikes_all
    ps = norm_spikes_by_label.T
    ps[neuron_active == False] = np.ones(n_classes) / n_classes
    return ps


# Takes array of probability distributions over labels & list of neuron indices.
# Returns array of probabability distributions over labels plus error class;
# Turns neurons in list indices into error neurons (ie. error probability == 1).
def add_error_class(ps, indices):
    n_e = ps.shape[0]
    n_classes = ps.shape[1]
    qs = np.zeros((n_e, n_classes+1))
    qs[:,:-1] = ps
    for i in indices:
        qs[i,:-1] = 0.
        qs[i,-1] = 1.
    return qs


##############################################################################
## Predict labels

# Takes 2d-array of probability distributions over labels and
# 2d-array of spike counts per input image.
# Returns a 2d-array of probability distributions over labels per input image.
def probability_predict(ps, spikes):
    assert(len(ps.shape) == 2 and len(spikes.shape) == 2)
    n_e, n_classes = ps.shape
    n_images, _n_e = spikes.shape
    assert(n_e == _n_e)
    norm_spikes, norm_exists = normalize_spikes(spikes)
    qs = np.zeros((n_images, n_classes))
    for i in range(n_images):
        qs[i] = np.matmul(norm_spikes[i], ps)
    return qs


# Returns the most likely class per distribution (ie. the mode of the dist).
def mode(qs):
    classes = np.argmax(qs, axis=1)
    classes[confidence(qs) == 0.] = -1
    return classes

# Returns the probability of the most likely class, per dist; higher is better.
def confidence(qs):
    return np.max(qs, axis=1)

# Returns the sum of the variance for each class, per dist; lower is better.
def sum_variance(qs):
    sum_var = np.zeros(len(qs))
    for i in range(len(sum_var)):
        sum_var[i] = np.sum(qs[i] * np.abs(1. - qs[i]))  # abs() to combat neg sign due to rounding errors
    return sum_var

# Takes an array of probabilities and returns a list corresponding to
# percentile 0, 10, 25, 50, 75, 90 and 100 of the non-zero probablities.
def percentiles(probabilities):
    nz_probs = probabilities[probabilities > 0.]
    return [np.percentile(nz_probs, q) for q in [0, 10, 25, 50, 75, 90, 100]]


##############################################################################
## Parse command line arguments

# Overrides constant labeling_file if argv contains a string (w/o spaces)
# starting with labeling= and ending with .npy
def parse_labeling_file():
    global labeling_file
    for arg in sys.argv[1:]:
        if arg.startswith('labeling=') and arg.endswith('.npy'):
            labeling_file = arg[len('labeling='):]
            return

# Overrides constant predict_file if argv contains a string (w/o spaces)
# starting with predict= and ending with .npy
def parse_predict_file():
    global predict_file
    for arg in sys.argv[1:]:
        if arg.startswith('predict=') and arg.endswith('.npy'):
            predict_file = arg[len('predict='):]
            return

# Calling the parse functions
if __name__ == '__main__':
    parse_labeling_file()
    parse_predict_file()


##############################################################################
## Load result file for labeling; extract spike counts and labels (last column)

def load_file(file):
    result = np.load(result_path + file)
    return result[:,:-1], result[:,-1]

# Loading spikes and labels for labeling neurons
print('LABELING: Loading ' + labeling_file)
spikes, labels = load_file(labeling_file)
assert(len(labels) > 0)  # labelling file must be non-empty

# number of neurons in labeling file
n_e = spikes.shape[1]


##############################################################################
## Parse some more command line arguments

# Overrides constant random_neurons if argv contains a list of neuron indices;
# the list must be written in Python syntax WITHOUT ANY SPACES.
def parse_random_neurons():
    global random_neurons
    for arg in sys.argv[1:]:
        try:
            list = json.loads(arg)
            if all(isinstance(i, int) and 0 <= i < n_e for i in list):
                random_neurons = list
                return
        except: pass

# Calling the parse functions
if __name__ == '__main__':
    parse_random_neurons()


##############################################################################
## Compute probability labeling

# 2d-array of probability distributions over labels, one per neuron.
# The probability distributions are over the set of classes in labels,
# plus an error class (last element in distribution vector).
labeling = add_error_class(probability_labels(spikes, labels), random_neurons)

# print distribution of labeling confidence
label_conf_dist = percentiles(confidence(labeling))
print('Labeling confidence, percentiles 0,10,25,50,75,90,100: ', end='')
print(', '.join(['{:.2f}'.format(p) for p in label_conf_dist]))

# Print probability labeling
def print_labeling(labeling, spikes):
    n_e, n_classes = labeling.shape
    rates = response_rates(spikes)
    neuron_active = rates > 0.
    classes = mode(labeling)
    probs = confidence(labeling)
    svars = sum_variance(labeling)
    print('Neuron,Response%,Mode,Probability,Variance', end='')
    for i in range(n_classes - 1):
        print(',p_' + str(i), end='')
    print(',p_error')
    for i in range(n_e):
        print('{:2d},{:4.1f}'.format(i, rates[i] * 100), end='')
        if neuron_active[i]:
            print(',{:d},{:.3f},{:.3f}'.format(classes[i], probs[i], svars[i]), end='')
        else:
            print(', ,     ,     ', end='')
        for j in range(n_classes):
            print(',{:.2f}'.format(labeling[i,j]), end='')
        print()

if show_labeling:
    print()
    print_labeling(labeling, spikes)
    print()


##############################################################################
## Load result file for prediction; extract spike counts and labels (last col)

# Loading spikes and labels for predicting digits
print('PREDICTION: Loading ' + predict_file)
spikes, labels = load_file(predict_file)
assert(len(labels) > 0)         # predict file must be non-empty
assert(spikes.shape[1] == n_e)  # neurons for labeling and prediction must match


##############################################################################
## Compute predicted probabilities based on labeling

# 2d-array of probability distributions over labels, one per image.
# The probability distributions are over the set of classes in labels,
# plus an error class (last element in distribution vector).
prediction = probability_predict(labeling, spikes)
predicted_class = mode(prediction)
_, prediction_exists = normalize_spikes(spikes)

# Print prediction probabilities
def print_prediction(prediction, spikes, labels):
    n_images, n_classes = prediction.shape
    classes = mode(prediction)
    _, prediction_exists = normalize_spikes(spikes)
    probs = confidence(prediction)
    svars = sum_variance(prediction)
    print('Image,Actual,Predicted,Probability,Variance', end='')
    for i in range(n_classes - 1):
        print(',p_' + str(i), end='')
    print(',p_error')
    for i in range(n_images):
        print('{:4d}'.format(i), end='')
        if 0 <= labels[i] < n_classes - 1:
            print(',{:d}.'.format(labels[i]), end='')
        else:
            print(',{:d}.'.format(n_classes - 1), end='')
        if prediction_exists[i]:
            print(',{:d},{:.3f},{:.3f}'.format(classes[i], probs[i], svars[i]), end='')
            for j in range(n_classes):
                print(',{:.2f}'.format(prediction[i,j]), end='')
        else:
            print(', ,     ,     ', end='')
            for j in range(n_classes):
                print(',    ', end='')
        print()

if show_prediction:
    print()
    print_prediction(prediction, spikes, labels)
    print()


##############################################################################
## Compute confusion matrix of prediction

# Returns the confusion matrix between actual and predicted for n_classes
# such that rows are actuals, columns are predicted.
# Classes outwith range(n_classes) are ignored.
def confusion_matrix(actual, predicted, n_classes):
    assert(actual.shape == predicted.shape)
    cm = np.zeros((n_classes, n_classes), dtype='int')
    for i in range(len(actual)):
        l = actual[i]
        p = predicted[i]
        if 0 <= l < n_classes and 0 <= p < n_classes:
            cm[l, p] += 1
    return cm


##############################################################################
## Print accuracy by class and overall

n_images, n_classes = prediction.shape

# merge all error labels into one error class (the last class)
adjusted_labels = np.copy(labels)
adjusted_labels[(labels >= n_classes - 1) | (labels < 0)] = n_classes - 1

# print the confusion matrix; last row/column reprensent error class
cm = confusion_matrix(adjusted_labels, predicted_class, n_classes)
print()
print(f'Confusion matrix for {np.sum(cm)}/{n_images} predictions')
print(cm)
print()

# print overall accuracy
overall_accuracy = len(np.where((predicted_class - adjusted_labels) == 0)[0]) / n_images
print('Accuracy overall: {:.3f}'.format(overall_accuracy))

# print distribution of prediction confidence
predict_conf_dist = percentiles(confidence(prediction))
print('Prediction confidence, percentiles 0,10,25,50,75,90,100: ', end='')
print(', '.join(['{:.2f}'.format(p) for p in predict_conf_dist]))

# Display confusion matrix graphically
if show_confusion:
    import matplotlib.pyplot as plt
    plt.matshow(cm, cmap='hot')
    plt.colorbar()
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks)
    plt.yticks(tick_marks)
    plt.ylabel('actual')
    plt.xlabel('predicted')
    plt.show()

print('END')
