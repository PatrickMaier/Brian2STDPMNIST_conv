# Smooth down-sampling convolutions based on 2x2 kernels

import numpy as np


# Compute a 2D smoothing convolution with a stride of 2.
# Input images must be 28x28.
# Returns a 14x14 matrix of convolved values.
def transform(image):
    out = np.zeros((14, 14))
    for yo in range(14):
        y = 2 * yo
        for xo in range(14):
            x = 2 * xo
            out[xo, yo] = (image[x:x+2, y:y+2]).sum() / 4
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

def show(image1, image2):
    _, ax = plt.subplots(1, 2)
    ax[0].imshow(image1, cmap='gray', vmin=0, vmax=255)
    ax[1].imshow(image2, cmap='gray', vmin=0, vmax=255)
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
    show(image, conv)
