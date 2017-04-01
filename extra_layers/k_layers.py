# -*- coding: utf-8 -*-
"""
Created on Tue May 24 17:39:39 2016

@author: phil

special layers for keras
"""

from keras import initializers,regularizers
from keras.engine import Layer
from keras import backend as K
import numpy as np
import theano
import theano.tensor as T
from utils import k_utils
theano.config.floatX='float32'



class gaussian2dMapLayer(Layer):
    ''' assumes 2-d  input
        Only Allows Square 
    '''
    def __init__(self,input_dim,init = 'zero',
                                 init_mean= None,
                                 init_sigma = None,
                                 scale = 1.0,
                               
                                 
                                 **kwargs):
        assert input_dim[0] == input_dim[1],"Input must be square"

        self.input_dim = input_dim
        self.scale = np.float(scale)
        self.inv_scale = self.input_dim[0]/self.scale
        
        xSpace = np.linspace(0,input_dim[0]-1,input_dim[0])
        ySpace = np.linspace(0,input_dim[1]-1,input_dim[1])
        spaceMatrix = np.asarray((np.meshgrid(xSpace,ySpace)))
        self.spaceVector = spaceMatrix.reshape((2,input_dim[0]*input_dim[1]))
        
        self.init = initializers.get(init) 
        
        if init_mean is None:
            half_mean = (1/2.)*self.scale
            init_mean = np.asarray([half_mean,half_mean])
          
        if init_sigma is None:
            one_sig = (np.asarray(1.0))*self.scale
            init_sigma =np.asarray([one_sig,
                                         np.asarray(0.0),
                                         one_sig])

        self.init_mean = init_mean.astype('float32')
        self.init_sigma = init_sigma.astype('float32')
        
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim)
        super(gaussian2dMapLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.mean = self.init((2,))

        self.sigma = self.init((3,))
        
        self.mean = self.add_weight((2,),
                                    initializer=self.init,
                                    name='mean')
        self.sigma = self.add_weight((3,),
                            initializer=self.init,
                            name='sigma')
        if self.init_mean is not None:
            self.mean.set_value(self.init_mean)
            del self.init_mean
        if self.init_sigma is not None:
            self.sigma.set_value(self.init_sigma)
            del self.init_sigma
        self.built = True

    def call(self,x, mask=None):
        x = T.reshape(x,(x.shape[0],x.shape[-2]*x.shape[-1]))
        tolerance = np.asarray(0.01) #use tolerance from stopping the matrix from being un-invertible
        T.set_subtensor(self.sigma[1] ,T.sgn(self.sigma[1])*T.min((T.sqrt(self.sigma[0]*self.sigma[2])-tolerance,T.abs_(self.sigma[1]))))

        inner = (self.spaceVector - self.inv_scale*self.mean.dimshuffle(0,'x'))

        cov = self.inv_scale*T.stack([[self.sigma[0],self.sigma[1]],[self.sigma[1],self.sigma[2]]])
        inverseCov = T.nlinalg.matrix_inverse(cov)
        firstProd =  T.tensordot(inner.T,inverseCov,axes=1)
        malahDistance = T.sum(firstProd*inner.T,axis =1)
        gaussianDistance = T.exp((-1./2.)*malahDistance)
        detCov = T.nlinalg.det(cov)
        denom = 1./(2*np.pi*T.sqrt(detCov))
        gdKernel = T.dot(x,denom*gaussianDistance)
        return gdKernel.dimshuffle(0,'x').astype('float32')
        


    def get_output_shape_for(self, input_shape):
        return (input_shape[0],1)
    def compute_output_shape(self, input_shape):
        return (input_shape[0],1)
    def get_config(self):
        base_config = super(gaussian2dMapLayer, self).get_config()
        return dict(list(base_config.items()))

        
