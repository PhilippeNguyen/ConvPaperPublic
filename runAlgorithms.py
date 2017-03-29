# -*- coding: utf-8 -*-
"""
Created on Tue May 17 12:29:05 2016

@author: phil
"""

from utils import p_utils
from curtis_lab_utils import clab_utils
from algorithms import p_algorithms
from algorithms import k_algorithms
import sys
import os
import scipy.signal
import numpy as np
from scipy.signal import gaussian
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from skimage.transform import resize
from models import k_allModelInfo

    


def runKRegression(stim,resp,options):
    print('runKRegression')
    
    myModel = k_allModelInfo.kRegression()
    stim = myModel._buildCalcAndScale(options,stim)
    
    X_train,X_reg,X_test,y_train,y_reg,y_test = clab_utils.splitDataSet(stim,resp)
    shapeFunc  = myModel.shapeStimulus()
    X_train,X_reg,X_test = [shapeFunc(x,options) for x in [X_train,X_reg,X_test]]
    options['Input_Shape'] =  myModel._getInputShape(X_train)

    #SGD is usually better than ADAM for the regression model
    result = k_algorithms.k_SIASGD(myModel,X_train,y_train,X_reg,y_reg,X_test,y_test,options)
    
    return result



def runKConvNet(stim,resp,options):
    print('runKConvNet')
        
    myModel = k_allModelInfo.kConvNet()
    stim = myModel._buildCalcAndScale(options,stim)

    X_train,X_reg,X_test,y_train,y_reg,y_test = clab_utils.splitDataSet(stim,resp)
    shapeFunc = myModel.shapeStimulus()
    X_train,X_reg, X_test = [shapeFunc(x,options) for x in [X_train,X_reg,X_test]]
    options['Input_Shape'] =  myModel._getInputShape(X_train)

    
    result = k_algorithms.k_SIAAdam(myModel,X_train,y_train,X_reg,y_reg,X_test,y_test,options)
    
    return result
def runKConvGaussNet(stim,resp,options):
    print('runKConvGaussNet')        
    
    myModel = k_allModelInfo.kConvGaussNet()
    stim = myModel._buildCalcAndScale(options,stim)

    X_train,X_reg,X_test,y_train,y_reg,y_test = clab_utils.splitDataSet(stim,resp)
    shapeFunc  = myModel.shapeStimulus()
    X_train,X_reg, X_test = [shapeFunc(x,options) for x in [X_train,X_reg,X_test]]
    options['Input_Shape'] =  myModel._getInputShape(X_train)


    result = k_algorithms.k_SIAAdam(myModel,X_train,y_train,X_reg,y_reg,X_test,y_test,options)
    
    return result
def runKConvDOGNet(stim,resp,options):
    print('runKConvDOGNet')
    
    myModel = k_allModelInfo.kConvDOGNet()
    stim = myModel._buildCalcAndScale(options,stim)

    X_train,X_reg,X_test,y_train,y_reg,y_test = clab_utils.splitDataSet(stim,resp)
    shapeFunc  = myModel.shapeStimulus()
    X_train,X_reg, X_test = [shapeFunc(x,options) for x in [X_train,X_reg,X_test]]
    options['Input_Shape'] =  myModel._getInputShape(X_train)

    result = k_algorithms.k_SIAAdam(myModel,X_train,y_train,X_reg,y_reg,X_test,y_test,options)
    
    return result
def runKConvGaussEXP(stim,resp,options):
    print('runKConvGaussEXP')        
    
    myModel = k_allModelInfo.kConvGaussEXP()
    stim = myModel._buildCalcAndScale(options,stim)

    X_train,X_reg,X_test,y_train,y_reg,y_test = clab_utils.splitDataSet(stim,resp)
    shapeFunc  = myModel.shapeStimulus()
    X_train,X_reg, X_test = [shapeFunc(x,options) for x in [X_train,X_reg,X_test]]
    options['Input_Shape'] =  myModel._getInputShape(X_train)

    result = k_algorithms.k_SIAAdam(myModel,X_train,y_train,X_reg,y_reg,X_test,y_test,options)
    
    return result
