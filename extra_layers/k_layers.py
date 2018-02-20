# -*- coding: utf-8 -*-
"""
Created on Tue May 24 17:39:39 2016

@author: phil

special layers for keras
"""
import numpy as np
import keras
from keras import initializers
from keras.engine import Layer
import keras.backend as K


if keras.backend._backend == 'theano':
    import theano
    import theano.tensor as T
    theano.config.floatX='float32'
    matrix_inverse = T.nlinalg.matrix_inverse
    tensordot = T.tensordot
    matrix_determinant = T.nlinalg.det
    
if keras.backend._backend == 'tensorflow':
    import tensorflow as tf
    matrix_inverse = tf.matrix_inverse
    tensordot = tf.tensordot
    matrix_determinant = tf.matrix_determinant
    
class maxoutLayer(Layer):
    def __init__(self, units,init_kernel=None,init_bias=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(maxoutLayer, self).__init__(**kwargs)
        self.units = units
        self.supports_masking = True
        self.init_kernel=None
        self.init_bias=None
#        if init_kernel is None:
##            self.init_kernel = np.expand_dims(np.linspace(-1,1,self.units).astype('float32'),
##                                              axis=0)
#            self.init_kernel = np.expand_dims(np.array([0.01,1]).astype('float32'),
#                                                    axis=0)
#        if init_bias is None:
#            self.init_bias  = np.zeros((1,self.units))

    def build(self, input_shape):
        print('hello')
        assert len(input_shape) >= 2

        self.kernel = self.add_weight(shape=(1,self.units),
                                      initializer='glorot_uniform',
                                      name='kernel')
        self.bias = self.add_weight(shape=(1,self.units),
                                    initializer='glorot_uniform',
                                    name='bias')
        if self.init_kernel is not None:
            print(self.init_kernel)
            K.set_value(self.kernel,self.init_kernel)
            del self.init_kernel
        if self.init_bias is not None:
            K.set_value(self.bias,self.init_bias)
            del self.init_bias

        self.built = True

    def call(self, inputs):
        input_shape = K.shape(inputs)
        inputs = K.batch_flatten(inputs)
        inputs = K.expand_dims(inputs,axis=-1)
        output = K.dot(inputs, self.kernel) + self.bias
        output = K.max(output,axis=-1)
        output = K.reshape(output,input_shape)
        
        
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'units': self.units,

        }
        base_config = super(maxoutLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
class gaussian2dMapLayer(Layer):
    ''' assumes 2-d  input
        Only Allows Square 
    '''
    def __init__(self,input_dim,init = 'zero',
                                 init_mean= None,
                                 init_sigma = None,                              
                                 
                                 **kwargs):
        assert input_dim[0] == input_dim[1],"Input must be square"

        self.input_dim = input_dim
        self.inv_scale = input_dim[0]
        
        #map the space of inputs, the values for the dot product will be pulled from a gaussian density
        xSpace = np.linspace(0,input_dim[0]-1,input_dim[0])
        ySpace = np.linspace(0,input_dim[1]-1,input_dim[1])
        spaceMatrix = np.asarray((np.meshgrid(xSpace,ySpace)))
        self.spaceVector = spaceMatrix.reshape((2,input_dim[0]*input_dim[1]))
        
        self.init = initializers.get(init) 
        
        if init_mean is None:
            half_mean = (1/2.)
            init_mean = np.asarray([half_mean,half_mean])
          
        if init_sigma is None:
            one_sig = (np.asarray(1.0))
            init_sigma =np.asarray([one_sig,
                                         np.asarray(0.0),
                                         one_sig])

        self.init_mean = init_mean.astype('float32')
        self.init_sigma = init_sigma.astype('float32')
        self.tolerance = np.asarray(0.01) #use tolerance from stopping the matrix from being un-invertible

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim)
        super(gaussian2dMapLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        
        self.mean = self.add_weight((2,),
                                    initializer=self.init,
                                    name='mean')
        self.sigma = self.add_weight((3,),
                            initializer=self.init,
                            name='sigma')
        if self.init_mean is not None:
            K.set_value(self.mean,self.init_mean)
            del self.init_mean
        if self.init_sigma is not None:
            K.set_value(self.sigma,self.init_sigma)
            del self.init_sigma
        self.built = True

    def call(self,x, mask=None):
        x = K.reshape(x,(K.shape(x)[0],K.shape(x)[-2]*K.shape(x)[-1]))
        covar = K.sign(self.sigma[1])*K.switch(K.sqrt(K.abs(self.sigma[0]*self.sigma[2]))-self.tolerance >= K.abs(self.sigma[1]),
                                                 K.abs(self.sigma[1]),
                                                 K.sqrt(K.abs(self.sigma[0]*self.sigma[2]))-self.tolerance )
#        covar = K.sign(self.sigma[1])*K.switch(K.sqrt(self.sigma[0]*self.sigma[2])-self.tolerance > K.abs(self.sigma[1]),
#                                                 K.abs(self.sigma[1]),
#                                                 K.sqrt(self.sigma[0]*self.sigma[2])-self.tolerance )

        #Below is just the calculations for a Gaussian
        inner = (self.spaceVector - self.inv_scale*K.expand_dims(self.mean))

        cov = self.inv_scale*K.stack([[self.sigma[0],covar],[covar,self.sigma[2]]])
        inverseCov = matrix_inverse(cov)
        firstProd =  tensordot(K.transpose(inner),inverseCov,axes=1)
        malahDistance = K.sum(firstProd*K.transpose(inner),axis =1)
        gaussianDistance = K.exp((-1./2.)*malahDistance)
        detCov = matrix_determinant(cov)
        denom = 1./(2*np.pi*K.sqrt(detCov))
        gdKernel = tensordot(x,denom*gaussianDistance,axes=1)
        return K.expand_dims(gdKernel)


    def get_output_shape_for(self, input_shape):
        return (input_shape[0],1)
    def compute_output_shape(self, input_shape):
        return (input_shape[0],1)
    def get_config(self):
        base_config = super(gaussian2dMapLayer, self).get_config()
        return dict(list(base_config.items()))

        
