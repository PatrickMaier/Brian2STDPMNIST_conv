# Source code for "Digit Classification using Biologically Plausible Neuromorphic Vision" [1]

In [1], we extend a previously studied spiking neural network (SNN) by Diehl and Cook [2] with convolutional filters that are sensitive to orientation. We investigate the impact of this extension on training behaviour (when learning to classify MNIST digits) as well as on robustness when confronted with unclassifiable inputs.

[1] examines 4 network variants by coupling the SNN of [2] with 4 different convolutional filter layers. These 4 variants are referred to as 28x28, 14x14, 28x28x2 and 14x14x2, respectively. In this document, the variable $NETWORK stands for one of these 4 variants. Their meaning is briefly explained below; we refer to [1] for details.

* 28x28 refers to a layer of 28x28 identity filters. That is, this variant is the same network as in [2].
* 14x14 refers to a layer of 14x14 smoothing filters.
* 28x28x2 refers to a layer of 28x28 horizontally and vertically oriented Prewitt filters.
* 14x14x2 refers to a layer of 14x14 horizontally and vertically oriented Prewitt filters.

The code in this repository is based on code [3] that was published by the authors of [2] and on the translation of that code [4] for the Brian 2 simulator.


## Prerequisites

The code requires the Brian 2 simulator. Some outputs require matplotlib.

```
pip install matplotlib
pip install brian2
pip install brian2tools
```


## Setup (data)

Training and testing requires the MNIST dataset.

1. Download the MNIST dataset from https://yann.lecun.com/exdb/mnist
2. Create a subdirectory mnist-data
3. Unpack the four *.gz files into subdirectory mnist-data
4. Run `python MNIST_init_data.py`

This will create files training.pickle and testing.pickle. After these files have been created, the MNIST dataset and the subdirectory mnist-data can be deleted.

Testing the robustness of the network wrt. unclassifiable inputs requires non-digit images. We use images from the CIFAR-100 dataset because their size is similar to the size of MNIST images.

1. Download the CIFAR-100 dataset from https://www.cs.toronto.edu/~kriz/cifar.html
2. Unpack cifar-100-python.tar.gz; this will create a subdirectory cifar-100-python
3. Run `python CIFAR_init_data.py`

This will pick 1000 CIFAR-100 images (balanced across all 100 classes), convert the images to grayscale, crop the central 28x28 pixels, and store the resulting 1000 test images in file cifar.pickle, in the same format as the MNIST images. After this file has been created, the CIFAR-100 dataset and the subdirectory `cifar-100-python` can be deleted.


## Setup (initial weights)

Prior to training, random initial weights need to be created, for each of the 4 network variants.

1. For each value of $NETWORK, run `python MNIST_init_weights.py $NETWORK`

This will create maps of random initial weights in subdirectory random/$NETWORK.


## Unsupervised training (epoch 1)

As [1] explains, training proceeds in two phase, the first of which is an unsupervised clustering of MNIST images. For a given $NETWORK, training the model for 1 epoch on the first 6000 MNIST training images is done using the following command.

```
python MNIST_train_model.py $NETWORK training.pickle 6000
```

Training instead on the third batch of 6000 training images is done by the following command (which skips 12000 training images before starting to train).

```
python MNIST_train_model.py $NETWORK training.pickle -12000 6000
```

Both commands will write the learned weights to file learned/$NETWORK/XeAe-6000.npy and the learned firing thresholds to learned/$NETWORK/theta_A-6000.npy. Every 1000 images, the commands will also dump intermediate learned weights and firing thresholds into the same subdirectory learned/$NETWORK. (The frequency of saving intermediate learned weights is governed by global variable save_interval in MNIST_train_model.py.)

In addition, the commands will also write a heat map visualisation of the learned weights to weights-fig/$NETWORK/XeAe-6000.png. Every 20 images, the commands will also dump intermediate visualisations of weights to subdirectory weights-fig/$NETWORK. (The frequency of writing intermediate visualisations is governed by global variable visual_interval in MNIST_train_model.py.)


## Unsupervised training (epoch 2)

In order to train for a second epoch, the learned weights and firing thresholds need to be copied to the respective weights subdirectory and renamed, as follows.

```
cp learned/$NETWORK/XeAe-6000.npy weights/$NETWORK/XeAe.npy
cp learned/$NETWORK/theta_A-6000.npy weights/$NETWORK/theta_A.npy
```

Then training for the second (and every subsequent) epoch is done using the same command as before, except for adding flag -pretrained, like so.

```
python MNIST_train_model.py $NETWORK -pretrained training.pickle 6000
```

Like before, this command will write the learned weights to file learned/$NETWORK/XeAe-6000.npy and the learned firing thresholds to learned/$NETWORK/theta_A-6000.npy. Intermediate learned weights and firing thresholds will be written as before, as will visualisations of weights. (Learned weights, firing thresholds and visualisations from previous epochs may be overwritten.)


## Supervised training (labeling)

The second training phase is a supervised labeling of output neurons. The trained model is run on the same set of training images, capturing spike count histograms indexed by the training images' labels (classes 0 to 9), for each output neuron. These histograms can be interpreted as a probability distribution for an output neuron's prediction.

Prior to labeling, the learned weights and firing thresholds need to be copied to the respective weights subdirectory and renamed, as before. Labeling using the first 6000 MNIST training images is done using the following command.

```
python MNIST_run_model.py $NETWORK training.pickle 6000
```

This command will write spike count histograms to file activity/$NETWORK/result_training.pickle_6000_[].npy.

Labeling using the third batch of 6000 MNIST training images (i.e. skipping 12000 images before labeling) is done as follows.

```
python MNIST_run_model.py $NETWORK training.pickle -12000 6000
```

This command will write spike count histograms to file activity/$NETWORK/result_training.pickle_-12000_6000_[].npy.


## Prediction (digits)

In order to do prediction, the trained model is run again, on test images, producing spike count histograms. These histograms are interpreted using the spike count histograms from the respective labeling run. For instance, to predict the first 1000 MNIST test images and evaluate the accuracy of the predictions, run the following commands (assuming weights and firing thresholds have been copied as outlined above).

```
python MNIST_run_model.py $NETWORK testing.pickle 1000
python MNIST_eval_runs.py $NETWORK labeling=result_training.pickle_6000_[].npy predict=result_testing.pickle_1000_[].npy
```

The MNIST_eval_runs.py script will print the labeling (i.e. the probability distribution) of each output neuron, the prediction confusion matrix, and the overall predication accuracy. (Probability distributions for each individual prediction can also be printed, by setting global variable show_prediction in MNIST_eval_runs.py accordingly.)

Example 1 below shows a full list of commands for training and testing.


## Prediction (non-digits)

In order to test robustness of the network when exposed to data that definitely aren't digits, we test using images derived from the CIFAR-100 dataset. The following commands assume that the data setup has been completed and that the model training and labeling phases have been completed. To test the networks predictions on 1000 non-digit images, use the following commands.

```
python MNIST_run_model.py $NETWORK cifar.pickle 1000
python MNIST_eval_runs.py $NETWORK labeling=result_training.pickle_6000_[].npy predict=result_cifar.pickle_1000_[].npy
```

The MNIST_eval_runs.py script will print the labeling, the confusion matrix and the overall accuracy. Since all 1000 images are non-digits, any classification shown by the confusion matrix is a misclassification. That is, only images that aren't classified (because no neuron spiked) are correctly recognised as non-digits.


## Prediction after resetting neurons

The models allow resetting of neurons prior to labeling and testing. In [1], we reset the 10 least confident neurons - which 10 neurons these are depends on the network and can be read off from the confidence of the labeling. Here we demonstrate how to reset neurons 0 to 9 during the labeling and testing phases.

```
python MNIST_run_model.py $NETWORK training.pickle 6000 [0,1,2,3,4,5,6,7,8,9]
python MNIST_run_model.py $NETWORK testing.pickle 1000 [0,1,2,3,4,5,6,7,8,9]
python MNIST_eval_runs.py $NETWORK labeling=result_training.pickle_6000_[0,1,2,3,4,5,6,7,8,9].npy predict=result_testing.pickle_1000_[0,1,2,3,4,5,6,7,8,9].npy [0,1,2,3,4,5,6,7,8,9]
```

It is important that the list of reset neurons is the same for the three commands, and that the list is passed as argument without spaces to the MNIST_run_model.py and MNIST_eval_runs.py scripts.

The MNIST_eval_runs.py script will print the labeling, confusion matrix and overall accuracy. Note that the confusion matrix includes an 11th class, the error class. In this particular case, no image should be classified in the error class, as all input images are digits. However, if the cifar.pickle file is used instead of testing.pickle, all images should be classified as error because none of those images resembles a digit.

Example 2 below shows a full list of commands for testing after resetting neurons.


## Example 1: Training the 14x14x2 network for 2 epochs on the 3rd batch of 6000 MNIST training images and testing on the 5th batch of 1000 MNIST testing images

We assume that the data setup and initial weight setup have been completed successfully.

```
# unsupervised clustering (epoch 1)
python MNIST_train_model.py 14x14x2 training.pickle -12000 6000
cp learned/14x14x2/XeAe-6000.npy weights/14x14x2/XeAe.npy
cp learned/14x14x2/theta_A-6000.npy weights/14x14x2/theta_A.npy
# unsupervised clustering (epoch 2)
python MNIST_train_model.py 14x14x2 -pretrained training.pickle -12000 6000
cp learned/14x14x2/XeAe-6000.npy weights/14x14x2/XeAe.npy
cp learned/14x14x2/theta_A-6000.npy weights/14x14x2/theta_A.npy
# supervised labeling
python MNIST_run_model.py 14x14x2 training.pickle -12000 6000
# prediction run
python MNIST_run_model.py 14x14x2 testing.pickle -4000 1000      
# prediction evaluation
python MNIST_eval_runs.py 14x14x2 labeling=result_training.pickle_-12000_6000_[].npy predict=result_testing.pickle_-4000_1000_[].npy
```


## Example 2: Resetting the 10 least confident neurons of the network trained in example 1 and testing on 1000 CIFAR-100 non-digit images

We assume that the weights of the model from the unsupervised clustering phases in example 1 have been copied as outlined above. We repeat the supervised labeling phase after resetting neurons 8, 14, 21, 22, 44, 46, 47, 56, 62, and 72. Then we predict, resetting the same list of neurons.

```
# supervised labeling (resetting 10 neurons)
python MNIST_run_model.py 14x14x2 training.pickle -12000 6000 [8,14,21,22,44,46,47,56,62,72]
# prediction run (resetting same 10 neurons)
python MNIST_run_model.py 14x14x2 cifar.pickle 1000 [8,14,21,22,44,46,47,56,62,72]
# prediction evaluation (resetting same 10 neurons)
python MNIST_eval_runs.py 14x14x2 labeling=result_training.pickle_-12000_6000_[8,14,21,22,44,46,47,56,62,72].npy predict=result_cifar.pickle_1000_[8,14,21,22,44,46,47,56,62,72].npy [8,14,21,22,44,46,47,56,62,72]
```


## References

1. Patrick Maier, James Rainey, Elena Gheorghiu, Kofi Appiah, and Deepayan Bhowmik. "Digit Classification using Biologically Plausible Neuromorphic Vision." SPIE Optics and Photonics, San Diego, August 2024.
2. Peter U. Diehl, and Matthew Cook. "Unsupervised learning of digit recognition using spike-timing-dependent plasticity". Frontiers Comput. Neurosci. 9, 99 (2015).
3. https://github.com/peter-u-diehl/stdp-mnist
4. https://github.com/zxzhijia/Brian2STDPMNIST
