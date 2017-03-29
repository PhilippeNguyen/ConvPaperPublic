# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 13:45:49 2016

@author: phil
"""
import keras
import theano
import numpy as np
from keras.layers import Dense, Dropout, Activation, Flatten,advanced_activations, Reshape
from keras.layers import Convolution2D, AveragePooling2D, Convolution1D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.regularizers import l1
from keras.layers.convolutional import conv_output_length
from extra_layers import k_layers
from keras.layers.core import ActivityRegularization

def buildkRegression(options):    
    numFeatures =  options['Input_Shape'][0]
    
    sum_L1_lambda = options['L1'] /(numFeatures)
    sum_L2_lambda = options['L2'] /(numFeatures) #not used
    regularizer = keras.regularizers.WeightRegularizer(l1=sum_L1_lambda, l2=sum_L2_lambda)

    
    #################
    #MODEL CREATION #
    #################
    model = Sequential()
    model.add(Dense(1,W_regularizer=regularizer,init=options['Dense_Init'],
                            weights = options['Initial_Weights'],input_dim=numFeatures))
    model.add(Activation('relu'))
    
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
    regularizer = keras.regularizers.WeightRegularizer(l1=sum_L1_lambda, l2=sum_L2_lambda)
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
                            
    model_prelu2 = k_layers.singlePReLU(weights=options['Initial_PReLU'])(model_conv1)
    model_pool3 = AveragePooling2D(pool_size=(options['Pool_Size'], options['Pool_Size']))(model_prelu2)

    model_flat4 = Flatten()(model_pool3)
    model_dense5 = Dense(1,W_regularizer=regularizer,init='glorot_normal',
                            weights = options['Initial_Map_Weights'])(model_flat4)
    output = Activation('relu')(model_dense5)
    activity_reg_output = ActivityRegularization(l1 =options['Activity_L1'],l2= options['Activity_L2'])(output)
    model = keras.models.Model(input=inputLayer,output =activity_reg_output)
#    
    return model

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
                            
    model_prelu2 = k_layers.singlePReLU(weights=options['Initial_PReLU'])(model_conv1)
    model_pool3 = AveragePooling2D(pool_size=(options['Pool_Size'], options['Pool_Size']))(model_prelu2)

    model_gaussian4 = k_layers.gaussian2dMapLayer(
        (downsampImageSize,downsampImageSize),
        init_mean=options['Initial_Gaussian_Mean'],
        init_sigma = options['Initial_Gaussian_Sigma'],
        init_sigma_div = options['Sigma_Div'],
        sigma_regularizer_l2 = keras.regularizers.l2(options['Sigma_Reg_L2']) )(model_pool3)
        
    model_dense5 = Dense((1),weights=options['Initial_Dense_Values'])(model_gaussian4)
    output = Activation('relu')(model_dense5)
    activity_reg_output = ActivityRegularization(l1 =options['Activity_L1'],l2= options['Activity_L2'])(output)

    model = keras.models.Model(input=inputLayer,output =activity_reg_output)
    
    return model

def buildkConvDOGNet(options):
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
    
    model_prelu2 = k_layers.singlePReLU(weights=options['Initial_PReLU'])(model_conv1)
    model_pool3 = AveragePooling2D(pool_size=(options['Pool_Size'], options['Pool_Size']))(model_prelu2)

    model_dog4 = k_layers.DOG2dMapLayer(
        (downsampImageSize,downsampImageSize),
        init_pos_mean = options['Initial_Gaussian_Pos_Mean'],
        init_pos_sigma = options['Initial_Gaussian_Pos_Sigma'],
        init_neg_mean = options['Initial_Gaussian_Neg_Mean'],
        init_neg_sigma = options['Initial_Gaussian_Neg_Sigma'],
        init_scale = options['Initial_DOG_Scale']
        )(model_pool3)
        
    model_dense5 = Dense((1),weights=options['Initial_Dense_Values'])(model_dog4)
    output = Activation('relu')(model_dense5)
    activity_reg_output = ActivityRegularization(l1 =options['Activity_L1'],l2= options['Activity_L2'])(output)

    model = keras.models.Model(input=inputLayer,output =activity_reg_output)
   
    
    return model
    
def buildkConvGaussEXP(options):
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
    inputDropout = keras.layers.Dropout(p =options['Input_Dropout'])(inputLayer)
    model_conv1 = Convolution2D(options['N_Kern'], options['Filter_Size'], options['Filter_Size'], 
                            border_mode='valid',
                            input_shape=options['Input_Shape'],
                            init='glorot_normal',
                            weights = options['Initial_Filter_Weights'],
                            subsample = stride)(inputDropout)
                            
    model_prelu2 = k_layers.singlePReLU(weights=options['Initial_PReLU'])(model_conv1)
    model_pool3 = AveragePooling2D(pool_size=(options['Pool_Size'], options['Pool_Size']))(model_prelu2)

    mapDropout = keras.layers.Dropout(p =options['Map_Dropout'])(model_pool3)

    model_gaussian4 = k_layers.gaussian2dMapLayerNormalized(
        (downsampImageSize,downsampImageSize),
        init_mean=options['Initial_Gaussian_Mean'],
        init_sigma = options['Initial_Gaussian_Sigma'],
        init_sigma_div = options['Sigma_Div'],
        scale = options['Gaussian_Layer_Scale'],
        sigma_regularizer_l2 = keras.regularizers.l2(options['Sigma_Reg_L2']) )(mapDropout)
        
    model_dense5 = Dense((1),weights=options['Initial_Dense_Values'])(model_gaussian4)
    output = keras.layers.advanced_activations.ParametricSoftplus()(model_dense5)
    activity_reg_output = ActivityRegularization(l1 =options['Activity_L1'],l2= options['Activity_L2'])(output)

    model = keras.models.Model(input=inputLayer,output =activity_reg_output)
    
    return model
    