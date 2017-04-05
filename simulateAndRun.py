#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import numpy as np
np.random.seed(10)  # this line controls the randomness, comment it out if you want unseeded randomness
import keras
keras.backend.set_image_dim_ordering('th')
from algorithms import k_algorithms
from models import k_allModelInfo

from scipy.io import loadmat

#for plotting
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from keras.utils.conv_utils import conv_output_length


def getNatStim():
    '''Loads the natural image dataset
    '''
    stim = loadmat('NaturalImages.mat')['stim']
    stim = stim.astype(np.float32)
    return stim
def getWhiteNoiseStim(movieSize):
    '''create white noise stimulus      
    '''    
    stim = np.random.normal(size=(movieSize))
    return stim
def main():
    
    options = dict()
    
    '''
    First, we generate a fake data. 
    Generateone simple cell with an LN model and 
    one complex cell using the Adelson-Bergen energy model. 
    
    For simplicity's sake, these models will have no time dynamics (though the convolutional
    model will include time dynamics)
    
    Note that the convolutional model 
    '''
  
    
    stimQuery = input('Use Natural Stimulus? White noise stimulus otherwise. (y/n) \n')
    if stimQuery =='y' or stimQuery=='yes':
        stim = getNatStim()
        movieSize = np.shape(stim)
        imSize = (movieSize[0],movieSize[1])
        movieLength = movieSize[2]
    else:
        #stimulus parameters (if using white noise)
        #you may require more data using white noise than natural images
        movieLength = 15000
        imSize = (30,30)
        movieSize = (*imSize,movieLength)
        stim = getWhiteNoiseStim(movieSize)

    #Set up neuron Gabors

    xCenter = 20.0
    yCenter= 10.0
    sf = 0.10
    ori = 30.0
    env = 2.0
    gabor0,gabor90 = generateGabors(imSize,xCenter,yCenter,sf,ori,env)
    
    #Plot the Gabors
    plt.imshow(gabor0)
    plt.title('Gabor 1')
    plt.waitforbuttonpress()
    plt.imshow(gabor90)
    plt.title('Gabor 2')
    plt.waitforbuttonpress()
    plt.close(plt.gcf())
    
    
    #get responses for gabors 
    
    gabor0Response = np.tensordot(gabor0,stim)
    gabor90Response = np.tensordot(gabor90,stim)
    

    
    #Generate cell responses and add noise
    noiseAmount = 0.1
    simpleCellResponse = np.maximum(0,gabor90Response + noiseAmount*np.random.normal(size=(movieLength,)))
    complexCellResponse = np.maximum(0,np.sqrt((gabor0Response**2) + (gabor90Response**2)) + noiseAmount*np.random.normal(size=(movieLength,)))
    
    #The code was initially set up to work alongside standard LN model estimation that was implemented in MATLAB
    #The movies must be converted into vectors (these will be reconverted into 2d
    #movies later). Note the Fortran order is needed
    #now the movie should be (numFrames,numPixelsPerFrame)
    options['Reshape_Order'] ='F'
    stim = np.reshape(stim,(imSize[0]*imSize[1],movieLength),order=options['Reshape_Order'] )
    stim = np.transpose(stim)
    
    #Finally we set up the estimation Set, regularization Set and prediction Set
    #use 80% for estimation, 10% for regularization and prediction each
    estIdx = slice(0,int(0.8*movieLength))
    regIdx = slice(int(0.8*movieLength),int(0.9*movieLength))
    predIdx = slice(int(0.9*movieLength),movieLength)
    estSet=stim[estIdx]
    regSet=stim[regIdx]
    predSet=stim[predIdx]
    
    
    '''
    Initialize the convolutional model
    '''
    
    myModel = k_allModelInfo.kConvGaussNet()
    
    #Choose model options (see k_buildModel and k_defaultOptions)
    
    options['Filter_Size'] = 11 #Here is an example option,
    options['Pool_Size'] = 3
    #We find that there are few important settings needed. The model works fine
    #with random initializations of the filter layer. However, if the filter is
    # too small, then the model will perform poorly
    
    #If an option isn't specified, then the value in k_defaultOptions will be used
    
    #set the amount of delays accounted for by the model (from t to t-8)
    #the optimal model would have options['Frames'] =list(range(1)), since there are no time dynamics
    options['Frames'] =list(range(8))

    
    #Generally, we want to normalize the stimulus, this is attached as part of 
    #the model
    estSet = myModel._buildCalcAndScale(options,estSet)
    regSet = myModel.stimScaler.applyScaleTransform(regSet)
    predSet = myModel.stimScaler.applyScaleTransform(predSet)
    
    shapeFunc  = myModel.shapeStimulus()
    estSet,regSet, predSet = [shapeFunc(x,options) for x in [estSet,regSet,predSet]]
    options['Input_Shape'] =  myModel._getInputShape(estSet)
    

    '''
    Estimate the simple cell
    '''
    y_est=simpleCellResponse[estIdx]
    y_reg=simpleCellResponse[regIdx]
    y_pred =simpleCellResponse[predIdx]
    
    simpleCellResults = k_algorithms.k_SIAAdam(myModel,estSet,y_est,regSet,y_reg,predSet,y_pred,options)

    '''
    Simple Cell Analysis
    '''
    simpleOpts = simpleCellResults['options']
    convImageSize = conv_output_length(imSize[0],simpleOpts['Filter_Size'],'valid',simpleOpts['Stride'])
    mapSize = conv_output_length(convImageSize,simpleOpts['Pool_Size'],'valid',simpleOpts['Pool_Size'])

    simpleCellWeights = simpleCellResults['model']['weights']
    filterWeights = simpleCellWeights[0]
    plotFilter(filterWeights)
    plt.title('Simple cell filter')
    plt.waitforbuttonpress()
    plt.close(plt.gcf())
    alpha = simpleCellWeights[2]
    plotAlpha(alpha)
    plt.title('Alpha value')
    plt.waitforbuttonpress()
    plt.close(plt.gcf())
    mapMean = simpleCellWeights[3]
    mapSigma = simpleCellWeights[4]
    mapVals = plotMap(mapMean,mapSigma,mapSize)
    plt.waitforbuttonpress()
    plt.close(plt.gcf())
    
    '''
    Estimate the complex cell
    Here we use the exact same model structure as the simple cell,though the weights will be re-initialized to random values
    '''
    y_est=complexCellResponse[estIdx]
    y_reg=complexCellResponse[regIdx]
    y_pred =complexCellResponse[predIdx]
    
    result = k_algorithms.k_SIAAdam(myModel,estSet,y_est,regSet,y_reg,predSet,y_pred,options)


    
    return

def generateGabors(imSize,xCenter,yCenter,sf,ori,env):
    ''' Very simple gabor parameterization '''
    #set up meshgrid for indexing
    x,y = np.meshgrid(range(imSize[0]),range(imSize[1]))
    a = 2*np.pi*np.cos(ori/(180.0/np.pi))*sf
    b = 2*np.pi*np.sin(ori/(180.0/np.pi))*sf    

    
    gauss = np.exp( - ((x-xCenter)**2+(y-yCenter)**2)/(2*(env**2)))
    sinFilt = np.sin(a*x+b*y)*gauss
    cosFilt = np.cos(a*x+b*y)*gauss
    
    return cosFilt/np.max(np.abs(cosFilt)),sinFilt/np.max(np.abs(sinFilt))


'''Plotting functions. 
    
'''

def plotFilter(filterWeights):
    numFrames = filterWeights.shape[2]
    for i in range(numFrames):
        plt.subplot(1,numFrames,i+1)
        plt.imshow(filterWeights[:,:,i,0])
        
        
    return
def plotAlpha(alpha):
    x = np.arange(-10,10)
    y = np.arange(-10,10)
    y[y<0] = alpha*y[y<0] 
    plt.plot(x,y)
    return
#Note: the map and reconstruction function assume that the 'scale' parameter
#       of the Gaussian Map is set to 1.0
def plotMap(mean,sigma,mapSize):
    sigmaVector = np.asarray([[sigma[0],sigma[1]],[sigma[1],sigma[2]]])
    
    meanVector = mean*mapSize
    sigmaVector = sigmaVector*mapSize
        
    x,y = np.meshgrid(range(mapSize),range(mapSize))
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    mvn = multivariate_normal(meanVector,sigmaVector) 
    myMap = mvn.pdf(pos)
    plt.imshow(myMap)
    return myMap
def plotReconstruction(filterWeights,mapWeights):
    pass


if __name__ == '__main__':
    main()
    