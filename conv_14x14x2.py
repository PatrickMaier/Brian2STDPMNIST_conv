# Edge detecting convolutions based on 3x3 kernels

import numpy as np


# Compute a 2D convolution with a stride of 2.
# Input images must be 28x28 and kernels must be 3x3.
# Returns a 14x14 matrix of convolved values (which may sometimes exceed 255)
def conv(image, kernel):
    image = np.pad(image, (0,1))
    out = np.zeros((14, 14))
    for yo in range(14):
        y = 2 * yo + 1
        for xo in range(14):
            x = 2 * xo + 1
            out[xo, yo] = max(0, (kernel * image[x-1:x+2, y-1:y+2]).sum()) / 3
    return out


# Vertical | kernel   
kernel_v = np.array([[+1,  0, -1],
                     [+1,  0, -1],
                     [+1,  0, -1]])

# Horizontal - kernel   
kernel_h = np.array([[+1, +1, +1],
                     [ 0,  0,  0],
                     [-1, -1, -1]])


# Compute 4 2D convolutions with a stride of 2, according to the above kernels.
# Input images must be 28x28.
# Returns a 28x28 matrix divided into quadrants such that each quadrant is
# the convolved image of a kernel (top row: vertical, horizontal;
# bottom row: inverted vertical, inverted horizontal).
def transform(image):
    out = np.zeros((28,28))
    out[ 0:14,  0:14] = conv(image,  kernel_v)  # vertical
    out[ 0:14, 14:28] = conv(image,  kernel_h)  # horizontal
    out[14:28,  0:14] = conv(image, -kernel_v)  # inverted vertical
    out[14:28, 14:28] = conv(image, -kernel_h)  # inverted horizontal
    return out


################################################################################
### TESTING

import os.path
import pickle
import matplotlib.pyplot as plt

def get_labeled_data(picklename):
    """Read input-vector (image) and target class (label, 0-9) and return
       it as list of tuples.
    """
    data = []
    if os.path.isfile('%s.pickle' % picklename):
        data = pickle.load(open('%s.pickle' % picklename, "rb"))
    return data

def show(image):
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.show()

# Process the i-th digit of the training (or testing) set
def test(i):
    data = get_labeled_data("training")
    #data = get_labeled_data("testing")

    image = data['x'][i]

    conv = transform(image)
    conv_pixels = conv.copy().reshape(-1).astype(int)
    conv_pixels.sort()
    print(conv_pixels.shape)
    print(conv_pixels[conv_pixels > 0])
    show(np.concatenate((image,conv)))
