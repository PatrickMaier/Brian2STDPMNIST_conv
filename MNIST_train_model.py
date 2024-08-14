### Rewrite of Diehl&Cook_spiking_MNIST_Brian2.py, with convolutional filters
###
### Simplified code; only trains the model.
### (I.e. functionality is similar to old code with test_mode = False)
###
### python MNIST_train_model.py NETWORK [-pretrained] [MNIST.pickle] [-SKIP] [N] [RESET]
### where
### * NETWORK is the string '28x28', '14x14', '28x28x2' or '14x14x2'
### * MNIST.pickle names a file of pickled MNIST (or other) training images
### * non-negative integer N limits the number of training images
### * negative integer -SKIP skips over the first abs(-SKIP) training images
### * list of neuron indices RESET indicates which neurons to reset to random
### * flag -pretrained loads pre-trained weights and firing thresholds

# Parse and import mandatory NETWORK parameter
import network_param
if __name__ == '__main__':
    network_param.parse()
from network_param import NETWORK

import json
import pickle
import time
import sys
import numpy as np
import brian2 as b2

# Import convolution layer, depending on NETWORK parameter
if NETWORK == '28x28':
    from conv_28x28 import transform
elif NETWORK == '14x14':
    from conv_14x14 import transform
elif NETWORK == '28x28x2':
    from conv_28x28x2 import transform
elif NETWORK == '14x14x2':
    from conv_14x14x2 import transform

# Imports from MNIST_model that Brian2 needs to run the simulation
from MNIST_model import \
    n_input, n_e, n_i, resting_time, single_image_time, \
    v_rest_e, v_rest_i, v_reset_e, v_reset_i, v_thresh_e, v_thresh_i, \
    v_offset_e, theta_plus_e, \
    refrac_e, refrac_i, tau_e, tau_i, \
    tc_theta, tc_pre_ee, tc_post_1_ee, tc_post_2_ee, \
    nu_ee_pre, nu_ee_post, wmax_ee, \
    neuron_eqs_e, neuron_eqs_i, \
    thresh_cond_e, thresh_cond_i, reset_act_e, reset_act_i, \
    model, pre_e, pre_i, post, \
    min_delay_input, max_delay_input, rdelay_input, \
    weight_ee_input, \
    NN

print('BEGIN')
np.random.seed(0)


##############################################################################
## Constants (some may be overridden by command line arguments)

data_path = './'      # path to all files and directories

image_file = 'training.pickle'  # default input images (MNIST train images)

#n_images = 6000       # default number of images to train model on
n_images = 10         # number of images to run model on

skip = 0              # number of images to skip over before training model

random_neurons = []   # list of indices of neurons w/ weights reset to random

pretrained = False    # start w/ pretrained weights and thetas if True

show_weights = False  # visualise input weights if True; requires matplotlib
#show_weights = True   # visualise input weights if True; requires matplotlib

#save_interval = 1000  # default interval for saving learned parameters
save_interval = 200   # interval for saving learned parameters

visual_interval = 20  # default interval for saving a visualisation of weights


##############################################################################
## Parse command line arguments

# Overrides constant pretrained if argv contains a flag -pretrained
def parse_pretrained():
    global pretrained
    for arg in sys.argv[1:]:
        if arg == '-pretrained':
            pretrained = True
            return

# Overrides constant image_file if argv contains a string ending in .pickle
def parse_image_file():
    global image_file
    for arg in sys.argv[1:]:
        if arg.endswith('.pickle'):
            image_file = arg
            return

# Overrides constant skip if argv contains a negative integer
def parse_skip():
    global skip
    for arg in sys.argv[1:]:
        try:
            n = int(arg)
            if n < 0:
                skip = -n
                return
        except: pass

# Overrides constant n_images if argv contains a non-negative integer
def parse_n_images():
    global n_images
    for arg in sys.argv[1:]:
        try:
            n = int(arg)
            if n >= 0:
                n_images = n
                return
        except: pass

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
    parse_pretrained()
    parse_image_file()
    parse_skip()
    parse_n_images()
    parse_random_neurons()


##############################################################################
## Data for initialising model parameters

# Load matrices of random weights
random_path = data_path + 'random/' + NETWORK + '/'
readout = np.load(random_path + 'AeAi.npy')
rweightAeAi_mat = np.zeros((n_e, n_i))
rweightAeAi_mat[np.int32(readout[:,0]), np.int32(readout[:,1])] = readout[:,2]
# NB: rweightAeAi_mat is a diagonal matrix; diagonal appears constant.

readout = np.load(random_path + 'AiAe.npy')
rweightAiAe_mat = np.zeros((n_i, n_e))
rweightAiAe_mat[np.int32(readout[:,0]), np.int32(readout[:,1])] = readout[:,2]
# NB: rweightAiAe_mat is a matrix with 0-diagonal; rest appears constant.

readout = np.load(random_path + 'XeAe.npy')
rweightXeAe_mat = np.zeros((n_input, n_e))
rweightXeAe_mat[np.int32(readout[:,0]), np.int32(readout[:,1])] = readout[:,2]
# NB: rweightXeAe_mat is a matrix of random weigths.

# Load pre-trained matrix of input weights (if pretrained && file available)
weight_path = data_path + 'weights/' + NETWORK + '/'
weightXeAe_mat = None
if pretrained:
    try:
        readout = np.load(weight_path + 'XeAe.npy')
        weightXeAe_mat = np.zeros((n_input, n_e))
        weightXeAe_mat[np.int32(readout[:,0]), np.int32(readout[:,1])] = readout[:,2]
    except:
        weightXeAe_mat = None

# Load pre-trained vector of firing thresholds (if pretrained && file available)
theta_vec = None
if pretrained:
    try:
        theta_vec = np.load(weight_path + 'theta_A.npy')
    except:
        theta_vec = None


##############################################################################
## Create neural network (in STDP mode)

nn = NN(True, rweightAeAi_mat, rweightAiAe_mat, rweightXeAe_mat, weightXeAe_mat, theta_vec)


##############################################################################
## Reset some neurons to random weights and default firing thresholds
## (governed by random_neurons list)

nn.reset_neurons(random_neurons, full_reset=True)


##############################################################################
## Loading input data

# load training images
training = pickle.load(open(data_path + image_file, 'rb'))
all_images = training['x']
all_labels = training['y'].T[0]

# rotate the first skip images to the end
if skip >= len(all_images):
    print('WARNING: Constant skip too large; skip set to 0')
    skip = 0
all_images = np.roll(all_images, -skip, axis=0)
all_labels = np.roll(all_labels, -skip, axis=0)

# guess whether dataset is MNIST (i.e. has exactly 10 classes)
training_is_MNIST = set(all_labels) == set(range(10))

# if it is MNIST data, make sure data is balanced
if training_is_MNIST:
    # select indices of first n_digits images of every digit
    n_digits = int(np.ceil(n_images / 10))
    indices = []
    for digit in range(10):
        indices += list(np.where(all_labels == digit)[0][:n_digits])

    # shuffle indices
    np.random.shuffle(indices)

    # restrict to images/labels corresponding to selected indices
    all_images = all_images[indices]
    all_labels = all_labels[indices]

# cut input data off at n_images
images = all_images[:n_images]
labels = all_labels[:n_images]
if len(images) < n_images:
    print('WARNING: Constant n_images too large; n_images set to', len(images))
    n_images = len(images)


##############################################################################
## Save input weights and firing thresholds (after training on k images)

def save_learned_params(k):
    out_path = data_path + 'learned/' + NETWORK + '/'
    end_fmt = '-%0' + str(len(str(n_images))) + 'd.npy'
    conn = nn.connections['XeAe']
    weight_triple_list = list(zip(conn.i, conn.j, conn.w))
    weights_filename = out_path + 'XeAe' + (end_fmt % k)
    np.save(weights_filename, weight_triple_list)
    theta_vec = nn.neuron_groups['Ae'].theta
    thresholds_filename = out_path + 'theta_A' + (end_fmt % k)
    np.save(thresholds_filename, theta_vec)


##############################################################################
## Visualise input weights (requires matplotlib)

def get_2d_input_weights():
    n_e_sqrt = int(np.ceil(np.sqrt(n_e)))
    n_input_sqrt = int(np.ceil(np.sqrt(n_input)))
    # reshape weights matrix; may work only if both dims are squares
    conn_w = np.copy(nn.connections['XeAe'].w)
    w = conn_w.reshape((n_input_sqrt, n_input_sqrt, n_e_sqrt, n_e_sqrt))
    # create a new output matrix
    w_out = np.zeros((n_input_sqrt * n_e_sqrt, n_input_sqrt * n_e_sqrt))
    for i in range(n_e_sqrt):
        for j in range(n_e_sqrt):
            # rearrange weights; NB i and j reversed to produce same image
            # ad Diehl&Cook during training
            w_out[i*n_input_sqrt:(i+1)*n_input_sqrt, j*n_input_sqrt:(j+1)*n_input_sqrt] = w[:,:,j,i]
    return w_out

# Visualise input weigts as as a square block matrix;
# neurons are displayed column by column, starting at the top left.
def save_2d_input_weights(k, display=False):
    n_input_sqrt = int(np.ceil(np.sqrt(n_input)))
    import matplotlib.pyplot as plt
    weights = get_2d_input_weights()
    fig, ax = plt.subplots()
    plt.imshow(weights, cmap='hot_r', interpolation='nearest', vmin = 0., vmax = 1.)
    ax.set_xticks([n_input_sqrt*i+n_input_sqrt//2 for i in range(10)], [str(i)+'x' for i in range(10)])
    ax.set_yticks([n_input_sqrt*i+n_input_sqrt//2 for i in range(10)], list(range(10)))
    fig_path = data_path + 'weights-fig/' + NETWORK + '/'
    end_fmt = '-%0' + str(len(str(n_images))) + 'd.png'
    fig_filename = fig_path + 'XeAe' + (end_fmt % k)
    fig.savefig(fig_filename, bbox_inches='tight')
    if display:
        plt.show()
    plt.close()


##############################################################################
## Normalize weights (to prevent runaway weight inflation)

def normalize_weights():
    conn = nn.connections['XeAe']
    len_source = len(conn.source)
    len_target = len(conn.target)
    connection = np.zeros((len_source, len_target))
    connection[conn.i, conn.j] = conn.w
    temp_connection = np.copy(connection)
    colSums = np.sum(temp_connection, axis=0)
    colFactors = weight_ee_input / colSums
    for j in range(n_e):
        temp_connection[:,j] *= colFactors[j]
    conn.w = temp_connection[conn.i, conn.j]


##############################################################################
## Running the simulation

# pick a big timestep for faster simulation
b2.defaultclock.dt = 1. * b2.ms

# setting up result monitor
result_monitor = np.zeros((n_images, n_e), dtype='int32')

# simulation loop
print('Starting simulation')
sim_start = time.time()
previous_spike_count = np.zeros(n_e, dtype='int32')
for j in range(n_images):
    # save weights and thresholds every so often
    if j % save_interval == 0 and j > 0:
        save_learned_params(j)
    
    # visualise weights every so often
    if j % visual_interval == 0:
        save_2d_input_weights(j)

    for input_intensity in range(2, 5):
        # calm network with all inputs set to 0 for the resting time
        nn.input_groups['Xe'].rates = 0 * b2.Hz
        nn.net.run(resting_time)
       
        # convert input images to spike rates
        spike_rates = transform(images[j,:,:]).reshape((n_input)) / 8. * input_intensity

        # feed spike rates to inputs to produce spike trains
        nn.input_groups['Xe'].rates = spike_rates * b2.Hz

        # run simulation on current image
        nn.net.run(single_image_time)

        # count spikes overall and per excitatory neuron
        current_spike_count = nn.spike_counters['Ae'].count - previous_spike_count
        current_spikes = np.sum(current_spike_count)
        previous_spike_count = np.copy(nn.spike_counters['Ae'].count)

        # print progress info
        print('{'+f'image:{j},n_images:{n_images},sensitivity:{input_intensity/8},spikes:{current_spikes}'+'}')

        # normalise weights
        normalize_weights()

        # record spikes if there were at least 2
        if current_spikes >= 2:
            result_monitor[j,:] = current_spike_count
            break
        else:
            pass  # repeat with increased intensity if there were < 2 spikes

# end of simulation loop
sim_end = time.time()
print(f'Ending simulation: {n_images} images in {round(sim_end - sim_start, 2)} seconds')


##############################################################################
## Saving the output

# save learned parameters (weights and firing thresholds)
save_learned_params(n_images)

# save a visualisation of the weights; display if show_weights == True
save_2d_input_weights(n_images, display=show_weights)

print('END')
