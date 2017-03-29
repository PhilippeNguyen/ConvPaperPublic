# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 16:23:41 2016

@author: phil
"""
from utils import p_utils
import numpy as np
from utils import k_utils


def minMax(X,given_axis = None,Xmin = 0,Xmax = 1):
    X_scaled = (X - Xmin) / (Xmax - Xmin)
    return X_scaled
def standardize(X, given_axis = None,given_mean = 0,given_std = 1):

    Xnorm = X - given_mean
    Xnorm = Xnorm / given_std
    Xnorm = np.nan_to_num(Xnorm)
    return Xnorm
def calcStandardize(X,given_axis = None):
    Xmean = np.mean(X,axis=given_axis)
    Xstd = np.std(X,axis=given_axis)
    return Xmean,Xstd
def calcMinMax(X,given_axis = None):
    Xmin = X.min(axis=given_axis)
    Xmax = X.max(axis=given_axis) 
    return Xmin,Xmax
    
    
class scaleStimulusClass(object):
    featureScalingParams = None
    axis = None
    scalingMethod = None
    def __init__(self,options):
        if 'Scaling_Method' in options:
            self.axis = options['Scaling_Method'][1]
            if options['Scaling_Method'][0] == 'Standardize':
                self.scalingCalcMethod = calcStandardize
                self.scalingMethod = standardize
            elif options['Scaling_Method'][0] == 'MinMax':
                self.scalingCalcMethod = calcMinMax
                self.scalingMethod = minMax
        else:
            self.scalingCalcMethod = calcStandardize
            self.scalingMethod = standardize
            self.axis = 0
        return
    def calcScaleParams(self,stim):
        self.featureScalingParams = (self.scalingCalcMethod(stim,self.axis))
        return self.featureScalingParams
    def applyScaleTransform(self,stim):
        assert self.featureScalingParams is not None
        stim = self.scalingMethod(stim,self.axis,*self.featureScalingParams)
        return stim

    
def kRegressionStyle(stim,options,trialSize = 375):
    ''' with standard a_movieClip, the movies are 375 frames long'''

    X = p_utils.dataDelay(stim,trialSize,options['Frames'])

    return X
def kConvNetStyle(stim,options,trialSize = 375):
    ''' with standard a_movieClip, the movies are 375 frames long'''    
    
    X = p_utils.dataDelay(stim,trialSize,options['Frames'])
    X = p_utils.dataDelayAsList(X,len(options['Frames']))
    #convert to (samples,frames,rows,cols)
    X = X.reshape(X.shape[0],X.shape[1],np.int(np.sqrt(X.shape[2])),np.int(np.sqrt(X.shape[2])),order=options['Reshape_Order'])
    
    return X
    
