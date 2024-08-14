### Rewrite of Diehl&Cook_MNIST_random_conn_generator.py
###
### Simplified code
###
### python MNIST_init_weights.py NETWORK
### where
### * NETWORK is the string '28x28', '14x14', '28x28x2' or '14x14x2'

# Parse and import mandatory NETWORK parameter
import network_param
if __name__ == '__main__':
    network_param.parse()
from network_param import NETWORK

import numpy as np

# Imports from MNIST_model
from MNIST_model import \
    n_input, n_e, n_i

print('BEGIN')
np.random.seed(0)


##############################################################################
## Constants

data_path = './'                                  # path to all files and dirs
out_path = data_path + 'random/' + NETWORK + '/'  # path to store init matrices

weight_XeAe_min = 0.01  # 'min' for XeAe weights (actual min is 0.01 * 0.3)
weight_XeAe_max = 0.3   # 'max' for XeAe weights (actual max is 1.01 * 0.3)
weight_AeAi = 10.4      # weight for AeAi diagonal
weight_AiAe = 17.0      # weight for AiAe off-diagonal


##############################################################################
## random XeAe matrix

XeAe = (np.random.random((n_input, n_e)) + weight_XeAe_min) * weight_XeAe_max
XeAeList = [(i, j, XeAe[i,j]) for i in range(n_input) for j in range(n_e)]
print('Creating XeAe matrix')
np.save(out_path + 'XeAe', XeAeList)


##############################################################################
## non-random AeAi square matrix (zero + constant diagonal)

assert(n_e == n_i)  # construction requires n_e == n_i
AeAiList = [(i, i, weight_AeAi) for i in range(n_e)]
print('Creating AeAi matrix')
np.save(out_path + 'AeAi', AeAiList)


##############################################################################
## non-random AiAe square matrix (constant + zero diagonal)

assert(n_e == n_i)  # construction requires n_e == n_i
AiAe = np.ones((n_i, n_e)) * weight_AiAe
for i in range(n_i):
    AiAe[i,i] = 0.0
AiAeList = [(i, j, AiAe[i,j]) for i in range(n_i) for j in range(n_e)]
print('Creating AiAe matrix')
np.save(out_path + 'AiAe', AiAeList)

print('END')
