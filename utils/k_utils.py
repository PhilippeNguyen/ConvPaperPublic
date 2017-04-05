# -*- coding: utf-8 -*-
"""
Created on Tue May 17 17:33:47 2016

@author: phil

"""

import numpy as np
from scipy.optimize import curve_fit
def conv_output_length(input_length, filter_size, border_mode, stride, dilation=1):
    
    #from keras 
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride
    
def SIAValidate(model,X_train,y_train,X_valid,y_valid,X_test,y_test,verbose = True):
    results = dict()
    estimatedModel = dict()
    
    #Pull estimated model parameters    
    estimatedModel['config'] = model.get_config()
    estimatedModel['weights'] = model.get_weights()   
    
    #Pull Validation results
    train_predictions =np.squeeze(model.predict(X_train))
    estimatedModel['p_opt'] = siaNLFit(y_train,train_predictions)   
    
    validation_predictions = np.squeeze(model.predict(X_valid))
    results['validVaf'],results['noNLValidVaf']  = siaNLVAF(y_valid,validation_predictions,estimatedModel['p_opt'])
    
    test_predictions = np.squeeze(model.predict(X_test))
    results['predVaf'],results['noNLPredVaf']  = siaNLVAF(y_test,test_predictions,estimatedModel['p_opt'])
    
    if verbose:
        printDictionary(results,'results')    
    
    return results,estimatedModel

def printDictionary(myDict,dictName = None):
    if dictName is not None:
        print('\n Printing {} ...'.format(dictName))
    for key,val in myDict.items():
        stringVal = str(val)
        if len(stringVal) < 50:
            print("{} = {}".format(key, stringVal))
        else:
           print("{} = {} ...".format(key, stringVal[:50]))
    return

def singletonSqueeze(x):
    view = x.dimshuffle([i for i in range(x.ndim) if not x.shape[i] ==1])
    return view

def vaf(y_true,y_pred):
    cc  = np.corrcoef(y_true,y_pred)
    vaf = np.square(cc[0,1])*100
    return vaf
    

def powerLaw(x, a, b):
        y = a*(x**b)
        return y
def powerLawFit(y_true,y_pred):
    #need to testThis
    p_opt,p_cov =curve_fit(powerLaw, np.float64(y_pred), np.float64(y_true),p0=[0.5,1.0])
    return p_opt
    
def siaNLFit(y_true,y_pred):
    #with SIA, we use a half wave rectifier NL then we fit a power law
    try:
        y_pred = np.maximum(0,y_pred)
        p_opt = powerLawFit(y_true,y_pred)
    except:
        print('error in siaNLFit')
        p_opt = [1.0,1.0]
    return p_opt
def siaNLPredict(y_pred,p_opt):
    y_pred =  np.maximum(0,y_pred)
    y_NL = powerLaw(y_pred,p_opt[0],p_opt[1])
    return y_NL
    
def siaNLVAF(y_true,y_pred,p_opt):
    noNL_VAF = vaf(y_true,y_pred)
    y_NL_pred = siaNLPredict(y_pred,p_opt)
    NL_VAF = vaf(y_true,y_NL_pred)
    
    return NL_VAF,noNL_VAF

def optDef(optionKey,optionDict,defaultVal):
    if optionKey in optionDict:
        optionVal  = optionDict[optionKey]
    else:        
        optionVal = defaultVal        
    return optionVal

def dataDelay(stim,trialSize,delay =[0]):
    ''' for every time point, new stim is the set of previous frames (delay ==0 is the current frame)
    Inputs: stim(m,n) array, m is the features, m is the examples
            trialSize, size of each trial, if you need a frame before the trial begins, it will be a zero-filled frame
            delay, array of delays to use. each delay corresponds to a previous input, ex delay = range(8), use all up to 7 preceding frames, 
            if delay = [0], new stim will be the same 
            if delay = [2], new stim will use only the stimulus from 2 frames ago'''
            
    stimSize = np.size(stim,axis=0)
    splitIndices = np.arange(0,stimSize-trialSize,trialSize)+trialSize
    splitList = np.split(stim,splitIndices,axis=0)
    

    #fill stim with zeros, to prepare for adding delays
    stim = np.zeros((stimSize,np.size(delay)*np.size(stim,axis=1)))
    for trialNum in range(len(splitList)):
        trial = splitList[trialNum]
        for frameNum in range(np.size(trial,axis=0)):
            stimFrame = []
            for k in delay:
                delayNum  = frameNum -k
                
                if delayNum < 0:
                    delayFrame = np.zeros(np.shape(trial[delayNum,:]))
                else:
                    delayFrame = trial[delayNum,:]  
                        
                stimFrame = np.concatenate((stimFrame,delayFrame))
            stim[frameNum+trialNum*trialSize,:] = stimFrame
    
    return stim
    
def dataDelayAsList(stim,numFrames):
    stim = np.split(stim,numFrames,axis= 1)
    stim = np.dstack(stim)
    stim = np.swapaxes(stim,1,2)
    return stim
def dataDelayAsStack(stim,numFrames):
    stim = np.split(stim,numFrames,axis= 1)
    stim = np.dstack(stim)
    stim = np.swapaxes(stim,1,2)
    return stim    