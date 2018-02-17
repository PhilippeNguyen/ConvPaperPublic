# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 15:49:31 2016

@author: phil
"""

from utils.k_utils import optDef
import numpy as np
def adamDefaults(given_options):
    options = dict()
    options['Batch_Size'] = optDef('Batch_Size',given_options,500)    
    options['Learning_Rate'] = optDef('Learning_Rate',given_options,0.001)
    options['N_Epochs'] = optDef('N_Epochs',given_options,700)
    options['Patience'] = optDef('Patience',given_options,50)
    return options

    
def kConvGaussNet(given_options):
    options = dict()
    options['N_Kern'] = optDef('N_Kern',given_options,1)
    options['Filter_Size'] = optDef('Filter_Size',given_options,13)
    options['Pool_Size'] = optDef('Pool_Size',given_options,2)
    options['Stride'] = optDef('Stride',given_options,1)
    options['Initial_Filter_Weights'] = optDef('Initial_Filter_Weights',given_options,None)
    options['Initial_PReLU']= optDef('Initial_PReLU',given_options,0.5)
    options['Initial_Gaussian_Mean']= optDef('Initial_Gaussian_Mean',given_options,None)
    options['Initial_Gaussian_Sigma']= optDef('Initial_Gaussian_Sigma',given_options,None)
    
    options['Initial_Dense_Values']= optDef('Initial_Dense_Values',given_options,[np.ones((1,1)),np.zeros((1))])
    
    options['Loss_Function']= optDef('Loss_Function',given_options,'mse')
    
    return options

    
def kConvNet(given_options):
    options = dict()
        
    options['N_Kern'] = optDef('N_Kern',given_options,1)
    options['Filter_Size'] = optDef('Filter_Size',given_options,13)
    options['Stride'] = optDef('Stride',given_options,1)
    options['Initial_Filter_Weights'] = optDef('Initial_Filter_Weights',given_options,None)

    options['Initial_PReLU']= optDef('Initial_PReLU',given_options,0.5)

    options['L1'] = optDef('L1',given_options,0)
    options['L2'] = optDef('L2',given_options,0)
    options['Pool_Size'] = optDef('Pool_Size',given_options,2)
    options['Initial_Map_Weights'] = optDef('Initial_Map_Weights',given_options,None)
    
    options['Loss_Function']= optDef('Loss_Function',given_options,'mse')
    return options

def kConvMaxGaussNet(given_options):
    options = dict()
    options['N_Kern'] = 1
    options['Filter_Size'] = 13
    options['Stride'] = 1
    options['Pool_Size'] = 2
    options['N_Kern'] = 1
    options['Num_Pieces'] = 5
    options['Loss_Function']= 'mse'
    options.update(given_options)
    return options