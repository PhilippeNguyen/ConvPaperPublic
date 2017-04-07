## Intro
Hi this is example code.
Just run python simulateAndRun.py
The code will generate two gabor filters, and use those to generate a simple cell and a complex cell. 
The code will also generate either white noise or pull natural image stimuli from the McGill Colour Image Database.
http://tabby.vision.mcgill.ca/ (the images have been downsampled)

Once the stimulus and neuron responses are generated, we estimate our model and then plot the results.
If an image appears, click on it to move forward.
I've commented throughout simulateAndRun so that you can follow what is occuring.
Most of the model building occurs in the k_* files. Look through them to see how the models are set up.


## Requirements

In order to run the code, you will need Python 3 (have not tested with Python 2). 
You will also need scipy, numpy, and matplotlib. 
You will need Keras 2.0 (the code will work with either theano or tensorflow backend).
https://keras.io/

As with all theano/tensorflow implementations, you will find that the code runs faster when running on the GPU.
If you have an NVIDIA card, the libraries CUDA and CUDNN will be useful.
https://developer.nvidia.com/cuda-zone
https://developer.nvidia.com/cudnn


## Notes

Since this is just example code, I don't plan to maintain much of this repository unless there is a bug.

As I built this example, Keras and tensorflow both introduced big updates (keras 2.0 / tensorflow 1.0). I've updated and tested the code for these versions.
Hopefully they won't introduce new breaking changes in the future.

The complex cell response is generated using the Adelson-Bergen energy model. Our method does not solve for this model exactly. 
However, this method still performs well for the AB model. One could generate a complex cell by using spatially separated identical simple cell subunits.
Our method should perform even better on this kind of complex cell.

