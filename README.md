# Source code for "Digit Classification using Biologically Plausible Neuromorphic Vision" [1]

In [1], we extend a previously studied spiking neural network (SNN) by Diehl and Cook [2] with convolutional filters that are sensitive to orientation. We investigate the impact of this extension on training behaviour (when learning to classify MNIST digits) as well as on robustness when confronted with unclassifiable inputs.

TODO: Explain NETWORK variants


The code in this repository is based on code [3] that was published by the authors of [2] and on the translation of that code [4] for the Brian 2 simulator.


## Prerequisites

The code requires Brian 2. Some outputs require matplotlib.

```
pip install matplotlib
pip install brian2
pip install brian2tools
```


## Setup (data)

Training and testing requires the MNIST dataset.

1. Donwnload MNIST dataset from https://yann.lecun.com/exdb/mnist
2. Create a subdirectory `mnist-data`
3. Unpack the four *.gz files into subdirectory `mnist-data`
4. Run `python MNIST_init_data.py`

This will create files training.pickle and testing.pickle. After these files have been created, the MNIST dataset and the subdirectory `mnist-data` can be deleted.

Testing the robustness of the network wrt. unclassifiable inputs requires non-digit images. We used images from the CIFAR-100 dataset because their size is similar to the size of MNIST images.

1. Donwload the CIFAR-100 dataset from https://www.cs.toronto.edu/~kriz/cifar.html
2. Unpack cifar-100-python.tar.gz
3. Run `python CIFAR_init_data.py`

This will pick 1000 CIFAR-100 images (balanced across all 100 classes), convert the images to grayscale, crop the central 28x28 pixels, and store the resulting 1000 test images file cifar.pickle in the same format as the MNIST images. After this file has been created, the CIFAR-100 dataset and the subdirectory `cifar-100-python` can be deleted.


## Setup (initial weights)

Prior to training, random initial weights need to be created, for each of the 4 network variants.

1. Run `python MNIST_init_weights.py NETWORK` where NETWORK is the string representing the network variant.

This will create maps of random initial weights in directory random/NETWORK.


## Unsupervised training (epoch 1)


## Unsupervised training (epoch 2)

## Supervised training (labeling)

## Prediction (digits)

## Prediciton (non-digits)


## References

1. Patrick Maier, James Rainey, Elena Gheorghiu, Kofi Appiah, and Deepayan Bhowmik. "Digit Classification using Biologically Plausible Neuromorphic Vision." SPIE Optics and Photonics, San Diego, August 2024.
2. Peter U. Diehl, and Matthew Cook. "Unsupervised learning of digit recognition using spike-timing-dependent plasticity". Frontiers Comput. Neurosci. 9, 99 (2015).
3. https://github.com/peter-u-diehl/stdp-mnist
4. https://github.com/zxzhijia/Brian2STDPMNIST
