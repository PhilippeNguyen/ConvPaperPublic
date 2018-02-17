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
import keras.backend as K

def buildkConvMaxGaussNet(Input_Shape=None,Filter_Size=None,Stride=None,
                          Pool_Size=None,N_Kern=None,Initial_Filter_Weights=None,
                          Num_Pieces=5,
                          Initial_Gaussian_Mean=None,Initial_Gaussian_Sigma=None,
                          Initial_Dense_Values=None,
                          **kwargs):
    numRows =  Input_Shape[1]
    numCols =  Input_Shape[2]
    assert numRows == numCols
    numFrames = Input_Shape[0]
    
    convImageSize = conv_output_length(numRows,Filter_Size,'valid',Stride)
    downsampImageSize = conv_output_length(convImageSize,Pool_Size,'valid',Pool_Size)
    stride = (Stride,Stride)
    
    #################
    #MODEL CREATION #
    #################
    inputLayer =keras.layers.Input(shape=Input_Shape)
    model_conv1 = Conv2D(N_Kern, (Filter_Size, Filter_Size), 
                            padding='valid',
                            input_shape=Input_Shape,
                            kernel_initializer='glorot_normal',
                            weights = Initial_Filter_Weights,
                            strides = stride)(inputLayer)

    maxout2 = k_layers.maxoutLayer(units=Num_Pieces)(model_conv1)
    model_pool3 = AveragePooling2D(pool_size=(Pool_Size, Pool_Size))(maxout2)

    model_gaussian4 = k_layers.gaussian2dMapLayer(
        (downsampImageSize,downsampImageSize),
        init_mean=Initial_Gaussian_Mean,
        init_sigma = Initial_Gaussian_Sigma)(model_pool3)
        
    model_dense5 = Dense((1),weights=Initial_Dense_Values)(model_gaussian4)
    output = Activation('relu')(model_dense5)

    model = keras.models.Model(inputs=inputLayer,outputs =output)
    
    return model

def buildkConvGaussNet(Input_Shape=None,Filter_Size=13,Stride=1,
                       Pool_Size=2,N_Kern=1,
                       Initial_Filter_Weights=None,
                       Initial_PReLU=0.5,
                       Initial_Gaussian_Mean=None,Initial_Gaussian_Sigma=None,
                       Initial_Dense_Values=None,
                       **kwargs):
    numRows =  Input_Shape[1]
    numCols =  Input_Shape[2]
    assert numRows == numCols
    numFrames = Input_Shape[0]
    
    convImageSize = conv_output_length(numRows,Filter_Size,'valid',Stride)
    downsampImageSize = conv_output_length(convImageSize,Pool_Size,'valid',Pool_Size)
    stride = (Stride,Stride)
    
    #################
    #MODEL CREATION #
    #################
    inputLayer =keras.layers.Input(shape=Input_Shape)
    model_conv1 = Conv2D(N_Kern, (Filter_Size, Filter_Size), 
                            padding='valid',
                            input_shape=Input_Shape,
                            kernel_initializer='glorot_normal',
                            weights = Initial_Filter_Weights,
                            strides = stride)(inputLayer)
    preluWeight = np.array(Initial_PReLU,ndmin=3)                       
    model_prelu2 = PReLU(weights=[preluWeight],
                         shared_axes=[1,2,3])(model_conv1)
    model_pool3 = AveragePooling2D(pool_size=(Pool_Size, Pool_Size))(model_prelu2)

    model_gaussian4 = k_layers.gaussian2dMapLayer(
        (downsampImageSize,downsampImageSize),
        init_mean=Initial_Gaussian_Mean,
        init_sigma = Initial_Gaussian_Sigma)(model_pool3)
        
    model_dense5 = Dense((1),weights=Initial_Dense_Values)(model_gaussian4)
    output = Activation('relu')(model_dense5)

    model = keras.models.Model(inputs=inputLayer,outputs =output)
    
    return model

def buildkConvNet(Input_Shape=None,Filter_Size=13,Stride=1,Pool_Size=2,
                  N_Kern=1,
                  L1=0,L2=0,Initial_Filter_Weights=None,
                  Initial_PReLU=0.5,
                  Initial_Map_Weights=None,
                  **kwargs
                  ):
    numRows =  Input_Shape[1]
    numCols =  Input_Shape[2]
    assert numRows == numCols
    numFrames = Input_Shape[0]
    #Variable Setup
    #since the lambdas were originally estimated as regularization
    #the mean weight value of the map layer, we convert it here so it is a regularizer 
    #on the sum of the weights in the map layer.
    convImageSize = conv_output_length(numRows,Filter_Size,'valid',Stride)
    downsampImageSize = conv_output_length(convImageSize,Pool_Size,'valid',Pool_Size)
    sum_L1_lambda = L1 /(downsampImageSize**2)
    sum_L2_lambda = L2 /(downsampImageSize**2) #not used
    regularizer = keras.regularizers.L1L2(l1=sum_L1_lambda, l2=sum_L2_lambda)
    stride = (Stride,Stride)
    
    #################
    #MODEL CREATION #
    #################
    
    inputLayer =keras.layers.Input(shape=Input_Shape)
    model_conv1 = Conv2D(N_Kern,(Filter_Size, Filter_Size), 
                            padding='valid',
                            input_shape=Input_Shape,
                            kernel_initializer='glorot_normal',
                            weights = Initial_Filter_Weights,
                            strides = stride)(inputLayer)
                            
    preluWeight = np.array(Initial_PReLU,ndmin=3)                       
    model_prelu2 = PReLU(weights=[preluWeight],
                         shared_axes=[1,2,3])(model_conv1)
    model_pool3 = AveragePooling2D(pool_size=(Pool_Size, Pool_Size))(model_prelu2)

    model_flat4 = Flatten()(model_pool3)
    model_dense5 = Dense(1,W_regularizer=regularizer,init='glorot_normal',
                            weights = Initial_Map_Weights)(model_flat4)
    output = Activation('relu')(model_dense5)
    model = keras.models.Model(input=inputLayer,output =output)
  
    return model