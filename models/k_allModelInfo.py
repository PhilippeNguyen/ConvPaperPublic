# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 14:49:08 2016

@author: phil

Module for information on all models. For a model to work in pySIA, 
it should match the abstract classes here.
Of course, you can build your own abstract classes outside of the these functions,
so long as they have the following functions (to work with the rest of the code)

Models need:

    Stimulus shaping function (shapeStimulus)
        Takes the XYT stim and turns it into something keras usable.
        This shape must be equal to the shape keras uses to build the model.
    
        (Input shape calculator (_getInputShape) is also needed, usually the input shape is the stimulus shape
        but with the 0th axis (nb_samples) removed. If this is untrue for the model (e.g. time-separable
        models), then you will need to give the correct _getInputShape function)
    
    Default parameters function (defaultOptions)
        Sets the default options for the model for any missing options. This should cover
        all options for the building function
    
    Building function (buildModel)
        Builds the model from a set of options
    
The k_Model class also includes methods for:

buildStimScaler (_buildCalcAndScale is similar but also applies the transform as well)
    This is a sub class that will scale the stimulus and store the parameters used for scaling.
    Similar to the sci-kit learn preprocessing classes
    
    As of right now (DEC232016), the stimScaler class uses the options
    options['Scaling_Method'] = ('Scaling_Method',axis)


"""
from models import k_buildModel
from models import k_defaultOptions
from models import k_shapeStimulus

class k_Model(object):
    model_name = 'abstractModel'
    input_shape = None
    stimScaler = None
    def __init__(self):
        return
    def defaultOptions(self,options = dict()):
        '''Fills in the options dictionary with default values if any are missing '''
        pass
    def shapeStimulus(self):
        '''Returns the function to shape the stimulus for this model '''
        pass 
    def buildModel(self,options):
        '''Builds a model given options, the options must have all the required parameters'''
        pass
    def buildStimScaler(self,options):
        ''' Builds a scaling object, this is the same for all models'''
        self.stimScaler = k_shapeStimulus.scaleStimulusClass(options)
        return
    def _buildCalcAndScale(self,options,stim):
        ''' Builds a scaling object, calculates the parameters for the scaler and applies the
            transform on the stim.'''
        self.stimScaler = k_shapeStimulus.scaleStimulusClass(options)
        self.stimScaler.calcScaleParams(stim)
        return self.stimScaler.applyScaleTransform(stim)
    def _getInputShape(self,X):
        ''' Default Keras shape has (nb_samples,featuredim_1,... featuredim_n
            Simply return the shape without nb_samples'''
        
        return (X.shape[1:])
    
    
class kConvGaussNet(k_Model):
    model_name = 'kConvGaussNet'
    def __init__(self):
        super(kConvGaussNet,self).__init__()
        return
    def defaultOptions(self,options = dict()):
        return k_defaultOptions.kConvGaussNet(options)
    def shapeStimulus(self):
        return k_shapeStimulus.kConvNetStyle
    def buildModel(self,options):
        return k_buildModel.buildkConvGaussNet(**options)
    
class kConvMaxGaussNet(k_Model):
    model_name = 'kConvMaxGaussNet'
    def __init__(self):
        super(kConvMaxGaussNet,self).__init__()
        return
    def defaultOptions(self,options = dict()):
        return k_defaultOptions.kConvMaxGaussNet(options)
    def shapeStimulus(self):
        return k_shapeStimulus.kConvNetStyle
    def buildModel(self,options):
        return k_buildModel.buildkConvMaxGaussNet(**options)
    
class kConvNet(k_Model):
    model_name = 'kConvNet'

    def __init__(self):
        super(kConvNet,self).__init__()
        return
    def defaultOptions(self,options = dict()):
        return k_defaultOptions.kConvNet(options)
    def shapeStimulus(self):
        return k_shapeStimulus.kConvNetStyle
    def buildModel(self,options):
        return k_buildModel.buildkConvNet(**options)