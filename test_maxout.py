
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import numpy as np
np.random.seed(10)  # this line controls the randomness, comment it out if you want unseeded randomness
import keras
import keras.backend as K
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
  
    
    print('Using natural images\n')
    stim = getNatStim()
    movieSize = np.shape(stim)
    imSize = (movieSize[0],movieSize[1])
    movieLength = movieSize[2]


    #Set up neuron Gabors

    xCenter = 20.0
    yCenter= 8.0
    sf = 0.10
    ori = 90.0
    env = 2.0
    gabor0,gabor90 = generateGabors(imSize,xCenter,yCenter,sf,ori,env)
    

    #get responses for gabors 
    
    gabor0Response = np.tensordot(gabor0,stim)
    gabor90Response = np.tensordot(gabor90,stim)
    
    
    #Generate cell responses and add noise
    noiseAmount = 0.5
    simpleCellResponse = np.maximum(0,gabor0Response + noiseAmount*np.random.normal(size=(movieLength,)))
    complexCellResponse = np.maximum(0,np.sqrt((gabor0Response**2) + (gabor90Response**2)) + noiseAmount*np.random.normal(size=(movieLength,)))
    
    #The code was initially set up to work alongside standard LN model estimation that was implemented in MATLAB
    #The movies must be converted into vectors (these will be reconverted into 2d
    #movies later). Plotting assumes fortran order because of this
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
    
    myModel = k_allModelInfo.kConvMaxGaussNet()

#    myModel = k_allModelInfo.kConvNet() #You can use the affine dense layer convolution model instead of Gaussian 
                                        # (will need to plot using the plotStandardMap instead of plotGaussMap)

    
    
    #Choose model options (see k_buildModel and k_defaultOptions)
    
    options['Filter_Size'] = 9 #Here is an example option,
    options['Pool_Size'] = 2
    #We find that there are few important settings needed. The model works fine
    #with random initializations of the filter layer. However, if the filter is
    # too small, then the model will perform poorly
    
    #If an option isn't specified, then the value in k_defaultOptions will be used
    
    #set the amount of delays accounted for by the model (from t to t-4)
    #the optimal model would have options['Frames'] =list(range(1)), since there are no time dynamics
    options['Frames'] =list(range(4))

    
    #Generally, we want to normalize the stimulus, this is attached as part of 
    #the model
    estSet = myModel._buildCalcAndScale(options,estSet)
    regSet = myModel.stimScaler.applyScaleTransform(regSet)
    predSet = myModel.stimScaler.applyScaleTransform(predSet)
    
    shapeFunc  = myModel.shapeStimulus()
    estSet,regSet, predSet = [shapeFunc(x,options) for x in [estSet,regSet,predSet]]
    options['Input_Shape'] =  myModel._getInputShape(estSet)
    

#    '''
#    Estimate the simple cell
#    '''
#    y_est=simpleCellResponse[estIdx]
#    y_reg=simpleCellResponse[regIdx]
#    y_pred =simpleCellResponse[predIdx]
#    
#    simpleCellResults,est_model = k_algorithms.k_SIAAdam(myModel,estSet,y_est,
#                                               regSet,y_reg,predSet,y_pred,
#                                               options)
#    conv_layer = est_model.get_layer(index=1)
#    conv_layer_output = K.function(est_model.inputs,[conv_layer.output])
#    conv_output_est = conv_layer_output([estSet])[0]
#    conv_min,conv_max = np.min(conv_output_est),np.max(conv_output_est)
#    
#    '''
#    Simple Cell Analysis. Just plotting the weights. Click on the images to move to the next one
#    '''
#    simpleOpts = simpleCellResults['options']
#    convImageSize = conv_output_length(imSize[0],simpleOpts['Filter_Size'],'valid',simpleOpts['Stride'])
#    mapSize = conv_output_length(convImageSize,simpleOpts['Pool_Size'],'valid',simpleOpts['Pool_Size'])
#
#    simpleCellWeights = simpleCellResults['model']['weights']
#    
#    maxout_lin = simpleCellWeights[2]
#    maxout_bias = simpleCellWeights[3]
#    alpha_space = np.expand_dims(np.linspace(conv_min,conv_max),axis=-1)
#    alpha_y = np.max((alpha_space*maxout_lin)+maxout_bias,axis=-1)
#    plt.plot(alpha_space,alpha_y)
    
    '''
    Estimate the complex cell
    Here we use the exact same model structure as the simple cell,though the weights will be re-initialized to random values
    '''
    y_est=complexCellResponse[estIdx]
    y_reg=complexCellResponse[regIdx]
    y_pred =complexCellResponse[predIdx]
    
    complexCellResults,est_model = k_algorithms.k_SIAAdam(myModel,estSet,y_est,regSet,y_reg,predSet,y_pred,options)

    conv_layer = est_model.get_layer(index=1)
    conv_layer_output = K.function(est_model.inputs,[conv_layer.output])
    conv_output_est = conv_layer_output([estSet])[0]
    conv_min,conv_max = np.min(conv_output_est),np.max(conv_output_est)
    '''
    Complex Cell Analysis. Just plotting the weights. Click on the images to move to the next one
    '''
    complexOpts = complexCellResults['options']
    convImageSize = conv_output_length(imSize[0],complexOpts['Filter_Size'],'valid',complexOpts['Stride'])
    mapSize = conv_output_length(convImageSize,complexOpts['Pool_Size'],'valid',complexOpts['Pool_Size'])

    complexCellWeights = complexCellResults['model']['weights']
    maxout_lin = complexCellWeights[2]
    maxout_bias = complexCellWeights[3]
    alpha_space = np.expand_dims(np.linspace(conv_min,conv_max),axis=-1)
    alpha_y = np.max((alpha_space*maxout_lin)+maxout_bias,axis=-1)
    plt.plot(alpha_space,alpha_y)
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
def checkRotation(filterWeights):
    #theano convolution will flip the filter, whereas tensorflow doesn't
    if keras.backend._backend == 'tensorflow':
        return filterWeights
    else:
        return np.rot90(filterWeights,k=2)
    
def plotFilter(filterWeights):
    numFrames = filterWeights.shape[2]
    vmin = np.min(filterWeights)
    vmax = np.max(filterWeights)
    for i in range(numFrames):
        plt.subplot(1,numFrames,i+1)
        plt.imshow(filterWeights[:,:,i],vmin=vmin,vmax=vmax)
        
        
    return
def plotAlpha(alpha):
    x = np.arange(-100,101)
    y = np.arange(-100,101)
    y[y<=0] = alpha*y[y<=0] 
    plt.plot(x,y)
    return

def plotGaussMap(mean,sigma,mapSize):
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

def plotStandardMap(mapWeights):
    mapSize = int(np.sqrt(np.shape(mapWeights)[0]))
    mapWeights = np.reshape(mapWeights,(mapSize,mapSize))
    
    plt.imshow(mapWeights)
    return mapWeights
def plotReconstruction(filterWeights,mapWeights,stride,poolSize,fullSize):
    mapSize = np.shape(mapWeights)[0]
    filterSize = np.shape(filterWeights)[0]
    numLags = np.shape(filterWeights)[2]
    unPoolFilter = np.zeros((mapSize*poolSize*stride,mapSize*poolSize*stride))
    reconFilter = np.zeros((fullSize,fullSize,numLags))
    for map_x_idx in range(mapSize):
        for map_y_idx in range(mapSize):

            unPoolFilter[(map_y_idx)*poolSize*stride:(map_y_idx+1)*poolSize*stride,(map_x_idx)*poolSize*stride:(map_x_idx+1)*poolSize*stride] = (
            unPoolFilter[(map_y_idx)*poolSize*stride:(map_y_idx+1)*poolSize*stride,(map_x_idx)*poolSize*stride:(map_x_idx+1)*poolSize*stride] +mapWeights[map_y_idx,map_x_idx])
    
    
    
    for lag in range(numLags):
        for x_idx in range(mapSize*poolSize*stride):
            for y_idx in range(mapSize*poolSize*stride):
                reconFilter[x_idx:x_idx+filterSize,y_idx:y_idx+filterSize,lag] = (
                        reconFilter[x_idx:x_idx+filterSize,y_idx:y_idx+filterSize,lag] +unPoolFilter[x_idx,y_idx]*filterWeights[:,:,lag])
    plotFilter(reconFilter)
    
    return reconFilter


if __name__ == '__main__':
    main()
    