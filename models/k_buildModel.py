# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 13:45:49 2016

@author: phil
"""
import keras
import numpy as np
from keras.layers import Dense, Dropout, Activation, Flatten,advanced_activations, Reshape
from keras.layers import Convolution2D, AveragePooling2D, Convolution1D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.regularizers import l1
from keras.utils.conv_utils import conv_output_length
from extra_layers import k_layers
from keras.layers.core import ActivityRegularization
from keras.layers.advanced_activations import PReLU



def buildkConvGaussNet(options):
    numRows =  options['Input_Shape'][1]
    numCols =  options['Input_Shape'][2]
    assert numRows == numCols
    numFrames = options['Input_Shape'][0]
    
    convImageSize = conv_output_length(numRows,options['Filter_Size'],'valid',options['Stride'])
    downsampImageSize = conv_output_length(convImageSize,options['Pool_Size'],'valid',options['Pool_Size'])
    stride = (options['Stride'],options['Stride'])
    
    #################
    #MODEL CREATION #
    #################
    inputLayer =keras.layers.Input(shape=options['Input_Shape'])
    model_conv1 = Convolution2D(options['N_Kern'], options['Filter_Size'], options['Filter_Size'], 
                            border_mode='valid',
                            input_shape=options['Input_Shape'],
                            init='glorot_normal',
                            weights = options['Initial_Filter_Weights'],
                            subsample = stride)(inputLayer)
    preluWeight = np.array(options['Initial_PReLU'],ndmin=3)                       
    model_prelu2 = PReLU(weights=[preluWeight],
                         shared_axes=[1,2,3])(model_conv1)
    model_pool3 = AveragePooling2D(pool_size=(options['Pool_Size'], options['Pool_Size']))(model_prelu2)

    model_gaussian4 = k_layers.gaussian2dMapLayer(
        (downsampImageSize,downsampImageSize),
        init_mean=options['Initial_Gaussian_Mean'],
        init_sigma = options['Initial_Gaussian_Sigma'])(model_pool3)
        
    model_dense5 = Dense((1),weights=options['Initial_Dense_Values'])(model_gaussian4)
    output = Activation('relu')(model_dense5)

    model = keras.models.Model(input=inputLayer,output =output)
    
    return model
    