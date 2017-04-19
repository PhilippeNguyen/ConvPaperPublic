## Intro
This repository contains Python code for a working example of our receptive field estimation method, using a simple convolutional neural network approach. The method is described in detail, with results on many cortical neurons, in our paper currently submitted for publication (“Estimating receptive fields of simple and complex cells in early visual cortex:  A convolutional neural network model with parameterized rectification”, by myself and Curtis Baker). This code generates responses of simulated model simple and complex V1 cells, to natural image as well as white noise stimuli. For each simulated neuron, it estimates the parameters of a receptive field model consisting of a single spatiotemporal filter, a parameterized rectified linear unit (PReLU), and a two-dimensional Gaussian "map" of the receptive field envelope.

## Description of the code
I've commented throughout the primary script, simulateAndRun.py, so that you can follow what is occuring.

The code generates two gabor filters, and uses those to create two model V1 neurons, to simulate a simple cell and a complex cell. Most of the model-building occurs in the k_* files. Look through them to see how the models are set up.

The code also generates either white noise stimuli, or uses natural image stimuli from the McGill Colour Image Database (http://tabby.vision.mcgill.ca/) - in the latter case, the images have been subdivided, converted to greyscale, and normalized (see Talebi & Baker, 2012), and downsampled.

Once the stimuli and neuron responses are generated, we estimate the model parameters, and then plot the results.

## Requirements
	Python 3 (have not tested with Python 2). 
	scipy, numpy, matplotlib 
	Keras 2.0 (the code will work with either theano or tensorflow backend) - https://keras.io/
As with all theano/tensorflow implementations, you will find that the code runs faster when running on the GPU.
If you have an NVIDIA card, the libraries CUDA and CUDNN will be useful.
	https://developer.nvidia.com/cuda-zone
	https://developer.nvidia.com/cudnn



## To run the code
Just run python simulateAndRun.py   As it executes, it will generate a series of images. When an image appears, click on it to move forward.


## Notes
Since this is just example code, I don't plan to maintain much of this repository unless there is a bug.
As I built this example, Keras and tensorflow both introduced big updates (keras 2.0 / tensorflow 1.0). I've updated and tested the code for these versions. Hopefully they won't introduce new breaking changes in the near future.

The complex cell response is generated using a simplified version of an "energy" model (Adelson & Bergen, 1975). Since our model architecture used for the estimation differs in some significant details from this energy-type model, our method does not exactly estimate its parameters. Nevertheless, our method still performs well on this model. One could generate a complex cell model by using spatially separated identical simple cell subunits, and replace the squaring operation with a full-wave rectification - in that case our method should perform even better.

Sorry, there is some mixing of terminology throughout, especially with regards to the three datasets. Here is a general outline (corresponding to training set/early stopping regularization set/ test set):  In the paper we use the terms Estimation/Regularization/Validation. Whereas in the code we use the terms Estimation/Validation/Prediction (though sometimes train/regularization/test is used). The model result will output the raw VAF on the validation set and raw VAF on the prediction set.

The final output non-linearity is not estimated in the keras framework. After the convolutional model is fitted, we take the model's predicted output to the training stimulus and the training set response, and use a half-power curve-fit. This is the predVAF and validVAFs. The noNL versions are the VAF computations without this last nonlinearity fitted. (However, you shouldn't see a difference between the noNL and "with NL" versions since the model does not have this nonlinearity).

	Note that the values of the Gaussian Map Layer correspond to the convolved image (the input to the gaussian map layer), and not image space itself.
