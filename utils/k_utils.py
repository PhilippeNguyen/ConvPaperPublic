# -*- coding: utf-8 -*-
"""
Created on Tue May 17 17:33:47 2016

@author: phil

utils for keras
"""

import numpy as np
from utils import p_utils
import scipy.signal
from multiprocessing import Pool
import matplotlib.pyplot as plt

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
    estimatedModel['p_opt'] = p_utils.siaNLFit(y_train,train_predictions)   
    
    validation_predictions = np.squeeze(model.predict(X_valid))
    results['validVaf'],results['noNLValidVaf']  = p_utils.siaNLVAF(y_valid,validation_predictions,estimatedModel['p_opt'])
    
    test_predictions = np.squeeze(model.predict(X_test))
    results['predVaf'],results['noNLPredVaf']  = p_utils.siaNLVAF(y_test,test_predictions,estimatedModel['p_opt'])
    
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
class plotGenerator(object):
    '''class to iterate over the pkl matDict to plot them'''
    def __init__(self,matDict,algorithm = None,closeEachIter = True):
        self.i = 0
        self.myDict = matDict
        self.keys = matDict.keys()
        self.length = len(matDict)
        self.algorithm = algorithm
        self.closeEachIter = True
        
    def next(self):
        if self.closeEachIter == True:
            plt.close('all')
        if self.i < self.length:
            
            neuron = self.myDict[self.keys[self.i]]
            plot_Neuron_Weights(neuron,self.algorithm)
            self.i +=1
        else:
            raise(StopIteration())
            
    def set_i(self,i):
        self.i = i

def plot_Neuron_Weights(neuronResults,algorithm = None):
    '''same as plot_kCNA_Weights but simply give it the
    neuron Data (list of results from all trials over all hyperparameters
    should be an entry(neuron) in the "bestDict"  '''

#    print(['neuronName: ' +str(neuronResults['neuronName'])])       
#    print(['predVaf: ' +str(neuronResults['predVaf'])])

    for field,vals in sorted(neuronResults.items(),key = lambda x:x[0]):
        if not isinstance(vals,dict):
            print([field +': ' +str(vals)])
    
    if algorithm == None:
        try:
            algorithm = neuronResults['method']
        except:
            print('no algorithm was given as parameter or no stored in the neuronResults')
    

    if algorithm == 'kConvNet' or algorithm == 'kOnOffConvNet':
        plot_kCNA_Weights(neuronResults)
    elif algorithm == 'kRadialNet':
        plot_radialProfile_weights(neuronResults)
    elif algorithm == 'kConvSplitNet':    
        plot_kConvSplit_Weights(neuronResults)
    else:
        print('plotting for this method is not implemented')
        
        
    return


def plot_kConvSplit_Weights(neuronResults):
    ''' this will assume the model is from the kConvSplitNet method, such that:
    weights[0] = filter weights
    weights[-2] = map weights '''
    filterWeights = neuronResults['model']['weights'][0]
    mapWeights = neuronResults['model']['weights'][-2]
    
    posWeights = mapWeights[:np.size(mapWeights,0)/2]
    negWeights = mapWeights[np.size(mapWeights,0)/2:]
    
    n_kern = neuronResults['options']['N_Kern']
    map_lags = 1
    
    #convert filterWeights to p_utils format
    #swap the nKern and NLags axes
    filterWeights  = np.swapaxes(filterWeights,0,1)
    p_utils.plotFilterWeights(filterWeights)
    
    #plot map weights, assumes map has 1 lag
    p_utils.plotMapWeights(posWeights,n_kern,map_lags,maxWeight = np.max(np.abs(mapWeights)))
    p_utils.plotMapWeights(negWeights,n_kern,map_lags,maxWeight = np.max(np.abs(mapWeights)))
    return    
    
def plot_kCNA_Weights(neuronResults):
    ''' this will assume the model is from the kCNA method, such that:
    weights[0] = filter weights
    weights[-2] = map weights '''
    filterWeights = neuronResults['model']['weights'][0]
    mapWeights = neuronResults['model']['weights'][-2]
    
    poolSize = neuronResults['options']['Pool_Size']
    filterSize = np.size(filterWeights,-1)
    mapSize = np.sqrt(np.size(mapWeights,0))
    
    n_kern = neuronResults['options']['N_Kern']
    map_lags = 1
    
    kernSize = (poolSize*mapSize -1) + filterSize 
    kernSize = np.int(np.ceil(np.float(kernSize)/poolSize)*poolSize)
    #convert filterWeights to p_utils format
    #swap the nKern and NLags axes
    filterWeights  = np.swapaxes(filterWeights,0,1)
    p_utils.plotFilterWeights(filterWeights)
    
    #plot map weights, assumes map has 1 lag
    p_utils.plotMapWeights(mapWeights,n_kern,map_lags)
    
    #plot reconstruction
    p_utils.plotReconstruction(mapWeights,filterWeights,downsamp= poolSize,kernsize = kernSize)  
    
    
    return

def plot_radialProfile_weights(neuronResults):
    ''' this will assume the model is from the kCNA method, such that:
    weights[0] = filter weights
    weights[-2] = map weights '''
    filterWeights = np.squeeze(neuronResults['model']['weights'][0])
    mapWeights = neuronResults['model']['weights'][-2]
    
    
    n_kern = neuronResults['options']['N_Kern']
    map_lags = 1    

    plt.figure()
    plt.plot(filterWeights)
    
    #plot map weights, assumes map has 1 lag
    p_utils.plotMapWeights(mapWeights,n_kern,map_lags)
    

    
    
    return

def createGaussian2dFilter(windowLength,std):
    oneDFilter = scipy.signal.gaussian(windowLength,np.float(std))/(windowLength)
    twoDFilter = np.outer(oneDFilter,oneDFilter)
    twoDFilter = np.expand_dims(twoDFilter,axis= 0)
    twoDFilter = np.expand_dims(twoDFilter,axis= 0)
    return twoDFilter
def createCosine2dFilter(windowLength):
    oneDFilter = scipy.signal.cosine(windowLength)/(windowLength)
    twoDFilter = np.outer(oneDFilter,oneDFilter)
    twoDFilter = np.expand_dims(twoDFilter,axis= 0)
    twoDFilter = np.expand_dims(twoDFilter,axis= 0)
    return twoDFilter

def conv2dToRadialProfile(movie,filterLength,stride = 1):
    '''movie format should be (frames,movieWindowLength(x-axis),movieWindowLength(y-axis))'''
    #mimic the 2d convolution, obtain the sub-images that would be multiplied by the filter kernel
    kernelLimit = np.int(np.size(movie,1) - filterLength)
    filterMid = int(filterLength/2)    
    pool = Pool(6)
    
    numSubImages = np.square(kernelLimit)
    subImageArray = []
    
    #just get the amount of values per radial profile
    y, x = np.indices((movie[0].shape))
    r = np.sqrt((x - filterMid)**2 + (y - filterMid)**2)
    radialProfileLength = len(np.bincount(r.astype(np.int).ravel()))
    
    
#    radialProfileArray =np.zeros((movie.shape[0],numSubImages,radialProfileLength))
#    arrayIndex = 0 
    for xStart in range(0,kernelLimit,stride):

        for yStart in range(0,kernelLimit,stride):
            
            subImage = movie[:, xStart:xStart+filterLength,yStart:yStart+filterLength]
#            radialValues = np.asarray([p_utils.radial_profile(subImage[frame,:,:],(filterMid,filterMid)) for frame in range(subImage.shape[0])])
#            radialProfileArray[:,arrayIndex,:] = radialValues
#            arrayIndex = arrayIndex +1 
            subImageArray = subImageArray + [subImage]
    
    
    subImageArray = (i for i in subImageArray)
    radialProfileArray = pool.imap(movieToRadialProfile,subImageArray)
    pool.close()
    pool.join()
    radialProfileArray  =  [i for i in radialProfileArray]
    radialProfileArray = np.swapaxes(np.asarray(radialProfileArray),axis1=1,axis2=0)
    
    
    return radialProfileArray

def movieToRadialProfile(movie):
    '''movie format should be (frames,movieWindowLength(x-axis),movieWindowLength(y-axis))'''
    z,y, x = movie.shape
    
    radialValues = np.asarray([p_utils.radial_profile(movie[frame,:,:],(float(x-1)/2,float(y-1)/2)) for frame in range(movie.shape[0])])
        
    return radialValues
    
#if __name__ == '__main__':
#    aa = mkCosTaper(256,0.3)
#    aa = combine(100,0,2,0)
    