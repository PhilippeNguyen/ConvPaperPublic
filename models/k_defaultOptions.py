# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 15:49:31 2016

@author: phil
"""

from utils.p_utils import optDef
import numpy as np
def adamDefaults(given_options):
    options = dict()
    options['Batch_Size'] = optDef('Batch_Size',given_options,500)    
    options['Learning_Rate'] = optDef('Learning_Rate',given_options,0.001)
    options['N_Epochs'] = optDef('N_Epochs',given_options,700)
    options['Patience'] = optDef('Patience',given_options,50)
    return options
def sgdDefaults(given_options):
    options = dict()    
    options['Batch_Size'] = optDef('Batch_Size',given_options,500)    
    options['Learning_Rate'] = optDef('Learning_Rate',given_options,0.001)
    options['N_Epochs'] = optDef('N_Epochs',given_options,700)
    options['Patience'] = optDef('Patience',given_options,50)
    options['Decay'] = optDef('Decay',given_options,0.0)
    options['Momentum'] = optDef('Momentum',given_options,0.0)    
    options['ClipNorm'] = optDef('ClipNorm',given_options,0.0)  
    options['ClipValue'] = optDef('ClipValue',given_options,0.0)  
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
    
    options['Activity_L1'] = optDef('Activity_L1',given_options,0.0)
    options['Activity_L2'] = optDef('Activity_L2',given_options,0.0)
    
    options['Loss_Function']= optDef('Loss_Function',given_options,'mse')
    return options
def kRegression(given_options):
    options = dict()
    options['Dense_Init'] = optDef('Dense_Init',given_options,'glorot_normal')
    options['Initial_Weights'] = optDef('Initial_Weights',given_options,None)
    options['L1'] = optDef('L1',given_options,0)
    options['L2'] = optDef('L2',given_options,0)
    
    options['Loss_Function']= optDef('Loss_Function',given_options,'mse')
    
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
    options['Sigma_Div']= optDef('Sigma_Div',given_options,1.0)
    options['Sigma_Reg_L2']= optDef('Sigma_Reg_L2',given_options,0.0)
    
    options['Initial_Dense_Values']= optDef('Initial_Dense_Values',given_options,[np.ones((1,1)),np.zeros((1))])
    
    options['Activity_L1'] = optDef('Activity_L1',given_options,0.0)
    options['Activity_L2'] = optDef('Activity_L2',given_options,0.0)
    
    options['Loss_Function']= optDef('Loss_Function',given_options,'msle')
    
    return options

def kConvDOGNet(given_options):
    options = dict()
    
    options['N_Kern'] = optDef('N_Kern',given_options,1)
    options['Filter_Size'] = optDef('Filter_Size',given_options,13)
    options['Pool_Size'] = optDef('Pool_Size',given_options,2)
    options['Stride'] = optDef('Stride',given_options,1)
    options['Initial_Filter_Weights'] = optDef('Initial_Filter_Weights',given_options,None)
    options['Initial_PReLU']= optDef('Initial_PReLU',given_options,0.5)
    options['Initial_Gaussian_Pos_Mean']= optDef('Initial_Gaussian_Pos_Mean',given_options,None)
    options['Initial_Gaussian_Pos_Sigma']= optDef('Initial_Gaussian_Pos_Sigma',given_options,None)
    options['Initial_Gaussian_Neg_Mean']= optDef('Initial_Gaussian_Neg_Mean',given_options,None)
    options['Initial_Gaussian_Neg_Sigma']= optDef('Initial_Gaussian_Neg_Sigma',given_options,None)
    options['Initial_DOG_Scale']= optDef('Initial_DOG_Scale',given_options,0.75)

    options['Initial_Dense_Values']= optDef('Initial_Dense_Values',given_options,[np.ones((1,1)),np.zeros((1))])
    
    options['Activity_L1'] = optDef('Activity_L1',given_options,0.0)
    options['Activity_L2'] = optDef('Activity_L2',given_options,0.0)
    
    options['Loss_Function']= optDef('Loss_Function',given_options,'msle')
    return options
    


def kConvGaussEXP(given_options):
    options = dict()
    options['Input_Dropout'] = optDef('Input_Dropout',given_options,0.0)
    options['N_Kern'] = optDef('N_Kern',given_options,1)
    options['Filter_Size'] = optDef('Filter_Size',given_options,13)
    options['Pool_Size'] = optDef('Pool_Size',given_options,2)
    options['Stride'] = optDef('Stride',given_options,1)
    options['Initial_Filter_Weights'] = optDef('Initial_Filter_Weights',given_options,None)
    options['Initial_PReLU']= optDef('Initial_PReLU',given_options,0.5)
    
    options['Map_Dropout']= optDef('Map_Dropout',given_options,0.0)
    
    options['Initial_Gaussian_Mean']= optDef('Initial_Gaussian_Mean',given_options,None)
    options['Initial_Gaussian_Sigma']= optDef('Initial_Gaussian_Sigma',given_options,None)
    options['Sigma_Div']= optDef('Sigma_Div',given_options,1.0)
    options['Sigma_Reg_L2']= optDef('Sigma_Reg_L2',given_options,0.0)
    options['Gaussian_Layer_Scale'] = optDef('Gaussian_Layer_Scale',given_options,1.0)
    
    options['Initial_Dense_Values']= optDef('Initial_Dense_Values',given_options,[np.ones((1,1)),np.zeros((1))])
    
    options['Activity_L1'] = optDef('Activity_L1',given_options,0.0)
    options['Activity_L2'] = optDef('Activity_L2',given_options,0.0)
    
    options['Loss_Function']= optDef('Loss_Function',given_options,'poisson')
    
    return options
