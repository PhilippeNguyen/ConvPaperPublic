#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import keras
keras.backend.set_image_dim_ordering('th')
import numpy as np
from algorithms import k_algorithms
from models import k_allModelInfo

#for plotting
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from keras.utils.conv_utils import conv_output_length

def main():
    
    options = dict()
    
    '''
    First, we generate a fake data, one simple cell with an LN model and one complex 
    cell using the Adelson-Bergen energy model.
    
    For simplicity's sake, these models will have no time dynamics (though the convolutional
    model will include time dynamics)
    '''
    movieLength = 10000
    imSize = (32,32)
    movieSize = (*imSize,movieLength)
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
    plt.gcf().clear()
    #create white noise stimulus (if you have natural image stimuli, you can
    #try it as well)
    
    wnStim = np.random.normal(size=(movieSize))
    
    #get responses for gabors
    gabor0Response = np.tensordot(gabor0,wnStim)
    gabor90Response = np.tensordot(gabor90,wnStim)
    

    
    #Generate cell responses
    simpleCellResponse = np.maximum(0,gabor0Response)
    complexCellResponse = np.sqrt((gabor0Response**2) + (gabor90Response**2))
    
    #The code was initially set up to work alongside standard LN model estimation
    #The movies must be converted into vectors (these will be reconverted into 2d
    #movies later). Note the Fortran order is needed
    #now the movie should be (numFrames,numPixelsPerFrame)
    options['Reshape_Order'] ='F'
    wnStim = np.reshape(wnStim,(imSize[0]*imSize[1],movieLength),order=options['Reshape_Order'] )
    wnStim = np.transpose(wnStim)
    
    #Finally we set up the estimation Set, regularization Set and prediction Set
    estIdx = slice(0,6000)
    regIdx = slice(6000,8000)
    predIdx = slice(8000,10000)
    estSet=wnStim[estIdx]
    regSet=wnStim[regIdx]
    predSet=wnStim[predIdx]
    
    
    '''
    Initialize the convolutional model
    '''
    
    myModel = k_allModelInfo.kConvGaussNet()
    
    #Choose model options (see k_buildModel and k_defaultOptions)
    
    options['Filter_Size'] = 11 #Here is an example option,
    #We find that there are few important settings needed. The model works fine
    #with random initializations of the filter layer. However, if the filter is
    # too small, then the model will perform poorly
    
    #If an option isn't specified, then the value in k_defaultOptions will be used
    
    #set the amount of delays accounted for by the model (from t to t-8)
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

    filterWeights = simpleCellResults['model']['weights'][0]
    plotFilter(filterWeights)
    plt.title('Simple cell filter')
    plt.waitforbuttonpress()
    plt.gcf().clear()
    mapMean = simpleCellResults['model']['weights'][3]
    mapSigma = simpleCellResults['model']['weights'][4]
    mapVals = plotMap(mapMean,mapSigma,mapSize)
    plt.waitforbuttonpress()

    
    '''
    Estimate the complex cell
    Here we use the exact same model structure as the simple cell,though the weights will be re-initialized to random values
    '''
    y_est=complexCellResponse[estIdx]
    y_reg=complexCellResponse[regIdx]
    y_pred =complexCellResponse[predIdx]
    
    result = k_algorithms.k_SIAAdam(myModel,estSet,y_est,regSet,y_reg,predSet,y_pred,options)


    
    pass

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

def plotFilter(filterWeights):
    numFrames = filterWeights.shape[2]
    for i in range(numFrames):
        plt.subplot(1,numFrames,i+1)
        plt.imshow(filterWeights[:,:,i,0])
        
        
    return
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
    