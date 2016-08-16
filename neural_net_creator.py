import numpy as np
import theano
import theano.tensor as T

import lasagne

class NeuralNetCreator:

    def __init__(self, input_shape, num_outputs, patch_size_array, pooling_size_array=[]
                    , num_filters=32,fully_con=[]):
        assert (len(pooling_size_array)==0 or len(pooling_size_array)==len(patch_size_array))
        self.input_shape=input_shape
        self.num_outputs=num_outputs
        self.patch_size_array=patch_size_array
        self.pooling_size_array=pooling_size_array
        self.num_filters=num_filters
        self.fully_con=fully_con
        self.buildNN()

    def buildNN(self):
        self.input_var = T.tensor4('inputs')
        self.target_var = T.ivector('targets')
        self.network = lasagne.layers.InputLayer(shape=self.input_shape, input_var=self.input_var)
        for patch_size_index in range(len(self.patch_size_array)):
            patch_size=self.patch_size_array[patch_size_index]
            pool_size = self.pooling_size_array[patch_size_index] if len(self.pooling_size_array)>0 else 2
            self.network = lasagne.layers.Conv2DLayer(self.network, num_filters=self.num_filters, 
                            filter_size=(patch_size, patch_size), 
                            nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
            self.network = lasagne.layers.MaxPool2DLayer(self.network, pool_size=(pool_size, pool_size))
        for num_units, dropout in self.fully_con[:-1]:
            self.network = lasagne.layers.DenseLayer(lasagne.layers.dropout(self.network, p=dropout),
                                num_units=num_units, nonlinearity=lasagne.nonlinearities.rectify)

        self.network = lasagne.layers.DenseLayer(lasagne.layers.dropout(self.network, p=self.fully_con[-1][1]),
                            num_units=self.fully_con[-1][0], nonlinearity=lasagne.nonlinearities.softmax)

