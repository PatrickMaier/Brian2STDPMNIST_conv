### Rewrite of Diehl&Cook_spiking_MNIST_Brian2.py
###
### Simplified code; only defines network and model/simulation constants;
### meant to be imported into simulation runs
###
### Network learns weights and firing thresholds iff NN() param STDP=True

from network_param import NETWORK
import numpy as np
import brian2 as b2

##############################################################################
## Model constants and equations (must be imported in full for Brian2 to run)

# network dimensions
if NETWORK == '28x28' or NETWORK == '14x14x2':
    n_input = 784   # features/image = number of input weights per exc neuron
elif NETWORK == '14x14':
    n_input = 196   # features/image = number of input weights per exc neuron
elif NETWORK == '28x28x2':
    n_input = 3136  # features/image = number of input weights per exc neuron
n_e = 100       # number of excitatory neurons
n_i = n_e       # number of inhibitory neurons

# simulation time
single_image_time = 0.35 * b2.second   # exposure time per image
resting_time = 0.15 * b2.second        # time between images

# potentials for excitatory and inhibitory neurons
v_rest_e = -65. * b2.mV
v_rest_i = -60. * b2.mV
v_reset_e = -65. * b2.mV
v_reset_i = -45. * b2.mV
v_thresh_e = -52. * b2.mV
v_thresh_i = -40. * b2.mV
v_offset_e = 20. * b2.mV    # firing threshold offset for excitatory neurons
theta_plus_e = .05 * b2.mV  # firing threshold increment

# refractory periods for excitatory and inhibitory neurons
refrac_e = 5. * b2.ms
refrac_i = 2. * b2.ms

# decay times for excitatory and inhibitory neurons
tau_e = 100. * b2.ms
tau_i =  10. * b2.ms

# (very long) decay time for adaptive firing thresholds
tc_theta = 1e7 * b2.ms

# decay times for STDP mode
tc_pre_ee = 20. * b2.ms
tc_post_1_ee = 20. * b2.ms
tc_post_2_ee = 40. * b2.ms

# learning rates etc for STDP mode
nu_ee_pre  = 0.001  # STDP learning rate (boosted 10x compared to Diehl&Cook)
nu_ee_post = 0.1    # STDP learning rate (boosted 10x compared to Diehl&Cook)
wmax_ee = 1.0       # weight saturation

# dynamics for excitatory neurons [NB: nS = nano Siemens, a unit of conductance]
#   v      : membrane voltage
#   ge     : conductance of excitatory synapses, decaying fast
#   gi     : conductance of inhibitory synapses, decaying not quite so fast
#   I_synE : current across excitatory synapses (depends on ge and v)
#   I_synI : current across inhibitory synapses (depends on gi and v)
#   timer  : timer for refractoriness period, advancing at 0.1x speed of clock
neuron_eqs_e = '''
    dv/dt = ((v_rest_e - v) + (I_synE/nS) + (I_synI/nS)) / tau_e : volt (unless refractory)
    I_synE = ge*nS * (-v)           : amp
    I_synI = gi*nS * (-100.*mV - v) : amp
    dge/dt = -ge / (1.*ms) : 1
    dgi/dt = -gi / (2.*ms) : 1
    dtimer/dt = 0.1 : second
    '''

# additional dynamics when not in STDP mode: learned firing threshold theta
neuron_eqs_e_theta = '''
    theta : volt
    '''

# additional dynamics when in STDP mode: exp decay of firing threshold theta
neuron_eqs_e_theta_STDP = '''
    dtheta/dt = -theta / (tc_theta) : volt
    '''

# dynamics for inhibitory neurons [NB: meaning of symbols as in neuron_eqs_e]
neuron_eqs_i = '''
    dv/dt = ((v_rest_i - v) + (I_synE/nS) + (I_synI/nS)) / tau_i : volt (unless refractory)
    I_synE = ge*nS * (-v)          : amp
    I_synI = gi*nS * (-85.*mV - v) : amp
    dge/dt = -ge / (1.*ms) : 1
    dgi/dt = -gi / (2.*ms) : 1
    '''

# threshold conditions for excitatory and inhibitory neurons
thresh_cond_e = '(v > (theta - v_offset_e + v_thresh_e)) and (timer > refrac_e)'
thresh_cond_i = 'v > v_thresh_i'

# reset actions for excitatory and inhibitory neurons
reset_act_e = 'v = v_reset_e; timer = 0.*ms'
reset_act_i = 'v = v_reset_i'

# additional reset actions for STDP mode: increase firing threshold theta
reset_act_e_STDP = '; theta += theta_plus_e'

# synapse dynamics
model = 'w : 1'         # syntaptic weights
pre_e = 'ge_post += w'  # effect of pre-synaptic excitatory spike
pre_i = 'gi_post += w'  # effect of pre-synaptic inhibitory spike
post = ''               # effect of post-synaptic spike

# additional synapse dynamics for STDP mode of Xe -> Ae connections
# NB: Unclear why there are two post variables, post1 and post2
model_STDP_ee = '''
    post2before                         : 1
    dpre/dt   = -pre / (tc_pre_ee)      : 1 (event-driven)
    dpost1/dt = -post1 / (tc_post_1_ee) : 1 (event-driven)
    dpost2/dt = -post2 / (tc_post_2_ee) : 1 (event-driven)
    '''

# additional actions on pre-synaptic Xe -> Ae spike in STDP mode
pre_STDP_ee = '; pre = 1.; w = clip(w + nu_ee_pre * post1, 0, wmax_ee)'

# additional actions on post-synaptic Xe -> Ae spike in STDP mode
post_STDP_ee = 'post2before = post2; w = clip(w + nu_ee_post * pre * post2before, 0, wmax_ee); post1 = 1.; post2 = 1.'

# random synaptic delay (only for input synapses)
min_delay_input =  0. * b2.ms
max_delay_input = 10. * b2.ms
rdelay_input = 'min_delay_input + rand() * (max_delay_input - min_delay_input)'

# constant used during weight normalisation (1/10 of n_input)
weight_ee_input = n_input / 10


##############################################################################
## Neural network

class NN:
    ## Constructor
    ## NB: If weightXeAe_mat or theta_vec are None, they are set to defaults.
    def __init__(self, \
                 STDP,             # True iff learning in STDP mode enabled \
                 rweightAeAi_mat,  # random weight matrix (AeAi synapses) \
                 rweightAiAe_mat,  # random weight matrix (AiAe synapses) \
                 rweightXeAe_mat,  # random weight matrix (XeAe synapses) \
                 weightXeAe_mat,   # learned weight matrix (XeAe synapses) \
                 theta_vec):       # learned firing threshold vector (e neurons)

        # memorised parameter (needed to reset neurons)
        self.rweightXeAe_mat = rweightXeAe_mat

        ######################################################################
        ### Neurons

        self.input_groups = {}
        self.neuron_groups = {}
        self.rate_monitors = {}
        self.spike_counters = {}

        # input/retina neurons creating spike trains
        self.input_groups['Xe'] = b2.PoissonGroup(n_input, 0 * b2.Hz)
        self.rate_monitors['Xe'] = b2.PopulationRateMonitor(self.input_groups['Xe'])

        # excitatory neurons
        if STDP:
            # STDP mode, only for excitatory neurons
            self.neuron_groups['e'] = b2.NeuronGroup(n_e, neuron_eqs_e+neuron_eqs_e_theta_STDP, threshold=thresh_cond_e, refractory=refrac_e, reset=reset_act_e+reset_act_e_STDP, method='exponential_euler')
        else:
            self.neuron_groups['e'] = b2.NeuronGroup(n_e, neuron_eqs_e+neuron_eqs_e_theta, threshold=thresh_cond_e, refractory=refrac_e, reset=reset_act_e, method='exponential_euler')
        if theta_vec is None:
            self.neuron_groups['e'].theta = np.ones((n_e)) * v_offset_e  # default firing thresholds
        else:
            self.neuron_groups['e'].theta = theta_vec * b2.volt          # learned firing thresholds

        # inhibitory neurons
        self.neuron_groups['i'] = b2.NeuronGroup(n_i, neuron_eqs_i, threshold=thresh_cond_i, refractory=refrac_i, reset=reset_act_i, method='exponential_euler')

        # subgroup of neuron_groups['e'] (all neurons, in fact)
        self.neuron_groups['Ae'] = self.neuron_groups['e'][0:n_e]
        self.neuron_groups['Ae'].v = v_rest_e - 40.*b2.mV   # initial membrane voltage
        self.rate_monitors['Ae'] = b2.PopulationRateMonitor(self.neuron_groups['Ae'])
        self.spike_counters['Ae'] = b2.SpikeMonitor(self.neuron_groups['Ae'])

        # subgroup of neuron_groups['i'] (all neurons, in fact)
        self.neuron_groups['Ai'] = self.neuron_groups['i'][0:n_i]
        self.neuron_groups['Ai'].v = v_rest_i - 40.*b2.mV   # initial membrane voltage
        self.rate_monitors['Ai'] = b2.PopulationRateMonitor(self.neuron_groups['Ai'])

        ######################################################################
        ### Synapses

        self.connections = {}

        # synapses input neurons -> excitatory neurons
        if STDP:
            # STDP mode, only for Xe -> Ae connections
            self.connections['XeAe'] = b2.Synapses(self.input_groups['Xe'], self.neuron_groups['Ae'], model=model+model_STDP_ee, on_pre=pre_e+pre_STDP_ee, on_post=post+post_STDP_ee)
        else:
            self.connections['XeAe'] = b2.Synapses(self.input_groups['Xe'], self.neuron_groups['Ae'], model=model, on_pre=pre_e, on_post=post)
        self.connections['XeAe'].connect(True)  # all-to-all connection
        self.connections['XeAe'].delay = rdelay_input  # random input delay [Q: why needed?]
        if weightXeAe_mat is None:
            self.connections['XeAe'].w = rweightXeAe_mat[self.connections['XeAe'].i, self.connections['XeAe'].j]  # random weights
        else:
            self.connections['XeAe'].w = weightXeAe_mat[self.connections['XeAe'].i, self.connections['XeAe'].j]   # learned weights

        # synapses excitatory neurons -> inhibitory neurons 
        self.connections['AeAi'] = b2.Synapses(self.neuron_groups['Ae'], self.neuron_groups['Ai'], model=model, on_pre=pre_e, on_post=post)
        self.connections['AeAi'].connect(True)  # all-to-all connection (reduced to 1-to-1 by diagonal weight matrix)
        self.connections['AeAi'].w = rweightAeAi_mat[self.connections['AeAi'].i, self.connections['AeAi'].j]

        # synapses inhibitory neurons -> excitatory neurons (feedback)
        self.connections['AiAe'] = b2.Synapses(self.neuron_groups['Ai'], self.neuron_groups['Ae'], model=model, on_pre=pre_i, on_post=post)
        self.connections['AiAe'].connect(True)  # all-to-all connection (slightly reduced by weight matrix with 0-diagonal)
        self.connections['AiAe'].w = rweightAiAe_mat[self.connections['AiAe'].i, self.connections['AiAe'].j]

        ######################################################################
        ### Building the network

        self.net = b2.Network()
        self.net.add(self.input_groups['Xe'])
        self.net.add(self.neuron_groups['e'])
        self.net.add(self.neuron_groups['Ae'])
        self.net.add(self.neuron_groups['i'])
        self.net.add(self.neuron_groups['Ai'])
        self.net.add(self.connections['XeAe'])
        self.net.add(self.connections['AeAi'])
        self.net.add(self.connections['AiAe'])
        self.net.add(self.rate_monitors['Xe'])
        self.net.add(self.rate_monitors['Ae'])
        self.net.add(self.rate_monitors['Ai'])
        self.net.add(self.spike_counters['Ae'])


    ##########################################################################
    ## Reset group neuron indices to random input weights.
    ## By default, firing thresholds are set to the population median;
    ## if opt arg full_reset==True, firing thresholds reset to initial default.
    def reset_neurons(self, indices, full_reset=False):
        if full_reset:
            theta_reset = v_offset_e
        else:
            theta_reset = np.median(self.neuron_groups['e'].theta)
        for i in indices:
            self.connections['XeAe'].w[:,i] = self.rweightXeAe_mat[:,i]
            self.neuron_groups['e'].theta[i] = theta_reset
