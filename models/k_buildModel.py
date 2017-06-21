# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 13:45:49 2016

@author: phil
"""
import keras
import numpy as np
from keras.layers import Dense, Activation,Flatten
from keras.layers import Conv2D, AveragePooling2D
from keras.utils.conv_utils import conv_output_length
from extra_layers import k_layers
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
    model_conv1 = Conv2D(options['N_Kern'], (options['Filter_Size'], options['Filter_Size']), 
                            padding='valid',
                            input_shape=options['Input_Shape'],
                            kernel_initializer='glorot_normal',
                            weights = options['Initial_Filter_Weights'],
                            strides = stride)(inputLayer)
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

    model = keras.models.Model(inputs=inputLayer,outputs =output)
    
    return model

def buildkConvNet(options):
    numRows =  options['Input_Shape'][1]
    numCols =  options['Input_Shape'][2]
    assert numRows == numCols
    numFrames = options['Input_Shape'][0]
    #Variable Setup
    #since the lambdas were originally estimated as regularization
    #the mean weight value of the map layer, we convert it here so it is a regularizer 
    #on the sum of the weights in the map layer.
    convImageSize = conv_output_length(numRows,options['Filter_Size'],'valid',options['Stride'])
    downsampImageSize = conv_output_length(convImageSize,options['Pool_Size'],'valid',options['Pool_Size'])
    sum_L1_lambda = options['L1'] /(downsampImageSize**2)
    sum_L2_lambda = options['L2'] /(downsampImageSize**2) #not used
    regularizer = keras.regularizers.L1L2(l1=sum_L1_lambda, l2=sum_L2_lambda)
    stride = (options['Stride'],options['Stride'])
    
    #################
    #MODEL CREATION #
    #################
    
    inputLayer =keras.layers.Input(shape=options['Input_Shape'])
    model_conv1 = Conv2D(options['N_Kern'],(options['Filter_Size'], options['Filter_Size']), 
                            padding='valid',
                            input_shape=options['Input_Shape'],
                            kernel_initializer='glorot_normal',
                            weights = options['Initial_Filter_Weights'],
                            strides = stride)(inputLayer)
                            
    preluWeight = np.array(options['Initial_PReLU'],ndmin=3)                       
    model_prelu2 = PReLU(weights=[preluWeight],
                         shared_axes=[1,2,3])(model_conv1)
    model_pool3 = AveragePooling2D(pool_size=(options['Pool_Size'], options['Pool_Size']))(model_prelu2)

    model_flat4 = Flatten()(model_pool3)
    model_dense5 = Dense(1,W_regularizer=regularizer,init='glorot_normal',
                            weights = options['Initial_Map_Weights'])(model_flat4)
    output = Activation('relu')(model_dense5)
    model = keras.models.Model(input=inputLayer,output =output)
#    
    return model