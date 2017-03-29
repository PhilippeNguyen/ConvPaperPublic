# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 18:45:22 2015

@author: phil
"""
import os
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
try:
    import pickle as pickle
except:
    import pickle
import sys
pySIALocation =  str(os.path.abspath(os.path.dirname(__file__)))
'''TODO: add padding function'''
from scipy.misc import imresize

def myInput(inputString):
    
    if sys.version_info[0] <3:
        val = raw_input(inputString)
    else:
        val = input(inputString)
    return val 
def myValues(myDict):
    if sys.version_info[0] <3:
        val = myDict.itervalues()
    else:
        val = myDict.values()
    return val  

def turnNantoZero(x):
    if np.isnan(x):
        x = 0
    return x
def userQuery(string,type,default,mode='user'):
    #modes are 'user': for user input, or 'default' automatically puts user input    
    
    
    if mode == 'user':
        if type == 'str':
            inpt = myInput(string + '[' + default +']? ')
            if inpt == '':
                output = default
            else:
                output = inpt
        elif type == 'int':
            inpt = myInput(string + '[' + str(default) +']? ')
            if inpt == '':
                output = default
            else:
                output = int(inpt)
        elif type == 'float':
            inpt = myInput(string + '[' + str(default) +']? ')
            if inpt == '':
                output = default
            else:
                output = float(inpt)
        elif type == 'list':
            inpt = myInput(string + '[ ' + str(default) +' ]? (list inputs seperated by spaces) ')
            if inpt == '':
                output = default
            else:
                output = list(map(int, inpt.split()))
    else:
        output = default

    
    return output

def optDef(optionKey,optionDict,defaultVal):
    if optionKey in optionDict:
        optionVal  = optionDict[optionKey]
    else:        
        optionVal = defaultVal        
    return optionVal
    

def load_shared_data(data_set):
    import theano
    #puts the data onto the gpu
    shared_set = theano.shared(np.asarray(data_set,
                                               dtype=theano.config.floatX),borrow=True)
    return shared_set
    
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
    
def dataPadd():
    return
def minMaxScale(x):
    minVal = np.min(x)
    maxVal =  np.max(x)
    x_std = (x - minVal) / (maxVal- minVal)
    return x_std

def radial_profile(data, center):
    ''' taken from http://stackoverflow.com/questions/21242011/most-efficient-way-to-calculate-radial-profile '''
    
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile[:int(center[0]+1)]
    
def plotMapWeights(weights,n_kern,nLags,transpose=True,maxWeight = None):
    #weights are 1-d
    if maxWeight == None:
        maxWeight = np.max(np.abs(weights))
    frameSize = (len(weights)/nLags)/n_kern
    kernSize = np.sqrt(frameSize)
    weights = np.reshape(weights,(nLags,n_kern,kernSize,kernSize))
    
    f, axarr = plt.subplots(n_kern, nLags)
    f.subplots_adjust(wspace=0.1)
    if n_kern == 1:
        axarr = np.expand_dims(axarr,axis=0)
    if nLags == 1:
        axarr = np.expand_dims(axarr,axis=1)
    
    for kern in range(n_kern):
        for lag in range(nLags):
            if transpose == True:
#                frame = np.rot90(weights[:,:,kern,lag],2)
                frame = np.transpose(weights[lag,kern,:,:])
            else:
                frame = weights[lag,kern,:,:]
                

            axarr[kern,lag]  .imshow(frame,cmap='gray',interpolation='none',clim=(-maxWeight,maxWeight))
            
            axarr[kern,lag]  .get_xaxis().set_visible(False)
            axarr[kern,lag]  .get_yaxis().set_visible(False)

        
    plt.show()
    return
def plotCNA(neuronResults,transpose=True):
    n_kern = neuronResults['options']['N_Kern']
    
    
    #plot filter
    #convert to 4-D np array 
    filterWeights = [np.squeeze(filterWeight,axis=1) for filterWeight in neuronResults['filterWeights']]
    filterWeights = np.asarray(filterWeights)
    plotFilterWeights(filterWeights,transpose = transpose)
    
    #plot map weights, assumes map has 1 lag
    plotMapWeights(neuronResults['mapWeights'],n_kern,1,transpose=transpose)
    
    #plot reconstruction
    plotReconstruction(neuronResults['mapWeights'],filterWeights,downsamp=neuronResults['options']['Pool_Size'],transpose = transpose)   
    
    return
        
def plotFilterWeights(weights,transpose=True,maxWeight = None):
    #weights are given as (nLags,n_kern,kernSize,kernSize)
    if maxWeight == None:
        maxWeight = np.max(np.abs(weights))
    nLags = np.size(weights,axis = 0)
    nKern = np.size(weights,axis = 1)
    f, axarr = plt.subplots(nKern, nLags)
    f.subplots_adjust(wspace=0.1)
    if nKern == 1:
        axarr = np.expand_dims(axarr,axis=0)
    if nLags == 1:
        axarr = np.expand_dims(axarr,axis=1)
        
    for lag in range(nLags):
        for kern in range(nKern):
            if transpose == True:
                frame = np.rot90(np.transpose(weights[lag,kern,:,:]),2)
            else:
                frame = weights[lag,kern,:,:]
                

            axarr[kern,lag]  .imshow(frame,cmap='gray',interpolation='none',clim=(-maxWeight,maxWeight))
            
            axarr[kern,lag]  .get_xaxis().set_visible(False)
            axarr[kern,lag]  .get_yaxis().set_visible(False)

            
    
    plt.show()
    return 
def plotReconstruction(mapWeights,filterWeights,kernsize,downsamp=3,mapLags=1,n_kern=1,transpose=True,plotFlag=True):
    
    frameSize = (len(mapWeights)/mapLags)/n_kern
    downsampleFrame_size = np.sqrt(frameSize)
    filterSize = np.size(filterWeights,axis = 3)
    mapWeights = np.reshape(mapWeights,(mapLags,n_kern,downsampleFrame_size,downsampleFrame_size))
    
    filterLags = np.size(filterWeights,axis = 0)
    if mapLags < filterLags:
        mapWeights = np.tile(mapWeights,(filterLags,1,1,1))
    
    reconFilter = np.zeros((filterLags,n_kern,kernsize,kernsize))    
    

    
    #recreate the filter
    for lag in range(filterLags):
        for kern in range(n_kern):
            if transpose == True:
#                frame = np.rot90(weights[:,:,kern,lag],2)
                mapFrame = np.transpose(mapWeights[lag,kern,:,:])
                filterFrame = np.rot90(np.transpose(filterWeights[lag,kern,:,:]),2)

            else:
                mapFrame = mapWeights[lag,kern,:,:]
                filterFrame = filterWeights[lag,kern,:,:]
                

            for x_idx in range(kernsize-filterSize+1):
                x_mapIdx = x_idx/downsamp
                for y_idx in range(kernsize-filterSize+1):
                    y_mapIdx = y_idx/downsamp
                    
                    if y_mapIdx < downsampleFrame_size and x_mapIdx < downsampleFrame_size:
                        reconFilter[lag,kern,y_idx:y_idx+filterSize,x_idx:x_idx+filterSize] = reconFilter[lag,kern,y_idx:y_idx+filterSize,x_idx:x_idx+filterSize] + filterFrame*mapFrame[y_mapIdx,x_mapIdx]
                
    #get the max weight, then plot 
    if plotFlag:
        maxWeight = np.max(np.abs(reconFilter))
        f, axarr = plt.subplots(n_kern, filterLags)
        f.subplots_adjust(wspace=0.1)
        if n_kern == 1:
            axarr = np.expand_dims(axarr,axis=0)
        if filterLags == 1:
            axarr = np.expand_dims(axarr,axis=1)
            
            
        for lag in range(filterLags):
            for kern in range(n_kern): 
                frame = reconFilter[lag,kern,:,:]
                axarr[kern,lag]  .imshow(frame,cmap='gray',interpolation='none',clim=(-maxWeight,maxWeight))
                
                axarr[kern,lag]  .get_xaxis().set_visible(False)
                axarr[kern,lag]  .get_yaxis().set_visible(False)
                
        
    
    return reconFilter
def upsample(mapWeights,filterSize,kernsize =30,downsamp=3,mapLags=1,n_kern=1,transpose=True):
    #assumes 1 kern for now!!!!!!!!!!!!!
    #purely upsampling, if you want to reverse the averaging, divide the output of this function by the square of downsamp
    frameSize = (len(mapWeights))
    downsampleFrame_size = np.int(np.sqrt(frameSize))

    mapWeights = np.reshape(mapWeights,(downsampleFrame_size,downsampleFrame_size))
    upMap = np.zeros((kernsize,kernsize))
    
    
    for x_idx in range(kernsize):
        x_mapIdx = x_idx/downsamp
        for y_idx in range(kernsize):
            y_mapIdx = y_idx/downsamp
            
            if y_mapIdx < downsampleFrame_size and x_mapIdx < downsampleFrame_size:
                upMap[y_idx:y_idx+filterSize,x_idx:x_idx+filterSize] = mapWeights[y_mapIdx,x_mapIdx]
        
    return np.reshape(upMap,(np.square(kernsize),1))
    


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

    

    


    



def estimateNoiseAdjVaf(respSet):
    #only works with the response sets given by getRespSet
    numReps = np.size(respSet,axis = 1)
    vafArray = np.zeros((numReps,))
    sumOfAllReps = np.sum(respSet,axis =1)
    for idx in range(numReps):
        thisRep = respSet[:,idx]
        sumOfOtherReps = sumOfAllReps - thisRep
        avgOfOtherReps = sumOfOtherReps / (numReps-1)
        thisVaf = vaf(thisRep,avgOfOtherReps)
        if np.isnan(thisVaf):
            thisVaf = 0
        vafArray[idx] = thisVaf
    totalVaf = np.mean(vafArray)
    return totalVaf

def estimateModelAdjVaf(modelPrediction,respSet):
    #only works with the response sets given by getRespSet
    numReps = np.size(respSet,axis = 1)
    vafArray = np.zeros((numReps,))
    for idx in range(numReps):
        thisRep = respSet[:,idx]
        thisVaf = vaf(thisRep,modelPrediction)
        if np.isnan(thisVaf):
            thisVaf = 0
        vafArray[idx] = thisVaf

    totalVaf = np.mean(vafArray)
    return totalVaf

def estimateNoiseCorr(respSet):
    #only works with the response sets given by getRespSet
    numReps = np.size(respSet,axis = 1)
    corrArray = np.zeros((numReps,))
    sumOfAllReps = np.sum(respSet,axis =1)
    for idx in range(numReps):
        thisRep = respSet[:,idx]
        sumOfOtherReps = sumOfAllReps - thisRep
        avgOfOtherReps = sumOfOtherReps / (numReps-1)
        thisCorr = np.corrcoef(thisRep,avgOfOtherReps)[0,1]
        if np.isnan(thisCorr):
            thisCorr = 0
        corrArray[idx] = thisCorr
    totalCorr = np.mean(corrArray)
    return totalCorr

def estimateModelCorr(modelPrediction,respSet):
    #only works with the response sets given by getRespSet
    numReps = np.size(respSet,axis = 1)
    corrArray = np.zeros((numReps,))
    for idx in range(numReps):
        thisRep = respSet[:,idx]
        thisCorr = np.corrcoef(thisRep,modelPrediction)[0,1]
        if np.isnan(thisCorr):
            thisCorr = 0
        corrArray[idx] = thisCorr

    totalCorr = np.mean(corrArray)
    return totalCorr
def estimateNeuronVariance(respSet):
    #only works with the response sets given by getRespSet
    #used in predictive power (as in Fregnac 2014)
    numReps = np.size(respSet,axis = 1)
    varArray = np.zeros((numReps,))
    sumOfAllReps = np.sum(respSet,axis =1)
    avgResponse = sumOfAllReps / (numReps)
    for idx in range(numReps):
        thisRep = respSet[:,idx]
        thisVariance = np.sum(np.square(thisRep -avgResponse))

        varArray[idx] = thisVariance
    totalVar = np.sum(varArray)
    return totalVar    

def estimateSSE(modelPrediction,respSet):
    #only works with the response sets given by getRespSet
    #used in predictive power (as in Fregnac 2014)
    numReps = np.size(respSet,axis = 1)
    sseArray = np.zeros((numReps,))
    for idx in range(numReps):
        thisRep = respSet[:,idx]
        thisSSE = np.sum(np.square(thisRep,modelPrediction))
        sseArray[idx] = thisSSE

    totalSSE = np.sum(sseArray)
    return totalSSE
    


def movieResize(movie,cropKern,interp,frameAxis):
    numFrames = np.size(movie,frameAxis)
    resizeMovie = np.zeros((cropKern,cropKern,numFrames))
    for movIdx in range(numFrames):
        resizeMovie[:,:,movIdx] = imresize(movie[:,:,movIdx],(cropKern,cropKern),interp = interp)
    
    return resizeMovie
def normalizeOverAllDim(X,featureMin = 0 ,featureMax= 1):
    X_ravel = np.ravel(X)
    X_std = (X - X_ravel.min(axis=0)) / (X_ravel.max(axis=0) - X_ravel.min(axis=0))
    X_scaled = X_std * (featureMax - featureMin) + featureMin
    return X_scaled

def normalize(X,featureMin = 0 ,featureMax= 1):
    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X_scaled = X_std * (featureMax - featureMin) + featureMin
    return X_scaled
def standardize(X):
    Xmean = np.mean(X,axis=0)
    Xstd = np.std(X,axis=0)
    Xnorm = X - Xmean
    Xnorm = Xnorm / Xstd
    Xnorm = np.nan_to_num(Xnorm)
    return Xnorm
  
#if __name__ == '__main__':
#    readStrfMatGUI()