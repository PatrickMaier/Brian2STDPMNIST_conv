### Extract and init/pickle MNIST test and train datasets
###
### To run:
### * Download MNIST dataset from https://yann.lecun.com/exdb/mnist/
### * Unpack the four *.gz into directory mnist-data
### * python MNIST_init_data.py
###   # Creates files training.pickle and testing.pickle in current directory

import numpy as np
import pickle
import struct

### Path constants
data_path = './'
MNIST_path = data_path + 'mnist-data/'  ## location of MNIST data set
out_path = data_path

### File name constants
train_image_fn = MNIST_path + 'train-images-idx3-ubyte'
train_label_fn = MNIST_path + 'train-labels-idx1-ubyte'
train_pickle_fn = out_path + 'training.pickle'
test_image_fn = MNIST_path + 't10k-images-idx3-ubyte'
test_label_fn = MNIST_path + 't10k-labels-idx1-ubyte'
test_pickle_fn = out_path + 'testing.pickle'


### Reads an MNIST image file and the corresponding label file.
### Returns dictionary of images (array x) and corresponding labels (array y).
def extract(image_filename, label_filename):
    with open(label_filename, 'rb') as lfile:
        # Get labels metadata
        assert(lfile.read(4) == b'\x00\x00\x08\x01')  # check magic number
        n, = struct.unpack('>I', lfile.read(4))
        # read labels
        y = np.zeros((n, 1), dtype=np.uint8)
        y[:] = list(struct.iter_unpack('>B', lfile.read(n)))
    
    with open(image_filename, 'rb') as ifile:
        # Get images metadata
        assert(ifile.read(4) == b'\x00\x00\x08\x03')  # check magic number
        n_images, rows, cols = struct.unpack('>III', ifile.read(3*4))
        assert(n == n_images)                         # check number of images
        # read images
        x = np.zeros((n, rows, cols), dtype=np.uint8)
        for i in range(n):
            pixels = list(struct.iter_unpack('>B', ifile.read(rows * cols)))
            x[i] = np.array(pixels, dtype=np.uint8).reshape(rows, cols)
    
    # return dictionary with extracted data
    return {'x': x, 'y': y, 'rows': rows, 'cols': cols}


### Read and pickle test dataset
print('Test dataset: reading ... ', end='', flush=True)
testing = extract(test_image_fn, test_label_fn)
print('pickling ... ', end='', flush=True)
pickle.dump(testing, open(test_pickle_fn, 'wb'))
print('done')

### Read and pickle training dataset
print('Train dataset: reading ... ', end='', flush=True)
training = extract(train_image_fn, train_label_fn)
print('pickling ... ', end='', flush=True)
pickle.dump(training, open(train_pickle_fn, 'wb'))
print('done')
