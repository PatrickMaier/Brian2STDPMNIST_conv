### Extract, convert and pickle some 1000 CIFAR-100 images as MNIST test dataset
###
### To run:
### * Download CIFAR-100 dataset: https://www.cs.toronto.edu/~kriz/cifar.html
### * Unpack cifar-100-python.tar.gz
### * python CIFAR_init_data.py
###   # Creates file cifar.pickle in current directory if it does not exist


import numpy as np
import pickle

### Path constants
data_path = './'
CIFAR_path = data_path + 'cifar-100-python/'  ## location of CIFAR-100 data set
out_path = data_path

### File name constants
train_fn = CIFAR_path + 'train'
test_fn = CIFAR_path + 'test'
meta_fn = CIFAR_path + 'meta'
pickle_fn = out_path + 'cifar.pickle'

### try loading cifar.pickle file
try:
    pickled_data = pickle.load(open(pickle_fn, 'rb'))
    print('pickled_data loaded')
except:
    pickled_data = None

### load CIFAR-100 data
print('CIFAR-100 loading ... ', end='', flush=True)
with open(meta_fn, 'rb') as f: meta = pickle.load(f, encoding='bytes')
with open(test_fn, 'rb') as f: test = pickle.load(f, encoding='bytes')
with open(train_fn, 'rb') as f: train = pickle.load(f, encoding='bytes')

### extract category names, numeric labels and image data
categories = meta[b'fine_label_names']
labels = np.asarray(train[b'fine_labels'] + test[b'fine_labels'], dtype='uint8')
images = np.concatenate((train[b'data'], test[b'data']))
n = len(images)

### convert images into 32x32 RGB pixels
rgb_images = np.transpose(images.reshape((n, 3, -1)), (0, 2, 1)).reshape((n, 32, 32, -1))

### convert RGB images to gray scale
gray_images = np.rint(0.3 * rgb_images[:,:,:,0] + 0.59 * rgb_images[:,:,:,1] + 0.11 * rgb_images[:,:,:,2]).astype('uint8')

### crop central 28x28 pixels from gray scale images
cropped_gray_images = gray_images[:,2:30,2:30]

### select indices of first 10 images of every category
indices = []
for l in range(100):
    indices += list(np.where(labels == l)[0][:10])
indices.sort()

### images and labels corresponding to selected indices
selected_labels = labels[indices]
selected_cropped_gray_images = cropped_gray_images[indices]

### adjustments to labels (to avoid overlap with MNIST labels)
selected_labels = selected_labels + 100  # shift labels up by +100
selected_labels = selected_labels.reshape((-1,1))  # match Diehl&Cook format

### assemble data (and check it's the same as the one loaded)
data = {'x': selected_cropped_gray_images, 'y': selected_labels, 'rows': 28, 'cols': 28}
if pickled_data != None:
    assert(data.keys() == pickled_data.keys())
    assert(data['cols'] == pickled_data['cols'])
    assert(data['rows'] == pickled_data['rows'])
    assert(np.array_equal(data['y'], pickled_data['y']))
    assert(np.array_equal(data['x'], pickled_data['x']))
print('done')

### pickle data (if pickle file didn't exist already)
if pickled_data == None:
    print('pickling data ... ', end='', flush=True)
    with open(pickle_fn, 'wb') as f: pickle.dump(data, f)
    print('done')


###############################################################################

# Function to plot individual images (for testing only - requires matplotlib)
# NB: Array indices stores the CIFAR-100 indices of the images.
#     To display the i-th image in the pickle file, run test(indices[i])
def test(i):
    import matplotlib.pyplot as plt
    print(i, labels[i], categories[labels[i]])
    plt.figure()
    plt.subplot(311)
    plt.imshow(rgb_images[i])
    plt.subplot(312)
    plt.imshow(gray_images[i], cmap='gray', vmin=0, vmax=255)
    plt.subplot(313)
    plt.imshow(cropped_gray_images[i], cmap='gray', vmin=0, vmax=255)
    plt.show()
