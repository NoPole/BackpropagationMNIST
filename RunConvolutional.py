#!/python


from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import theano
import theano.tensor as T

import Convolutional
from Convolutional import sigmoid, ReLU, tanh, Network
from Convolutional import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer


mini_batch_size = 10

   
   
def basic(n=3, epochs=20):
    
    for j in range(n):
        net = Network([
            ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                        filter_shape=(20, 1, 5, 5),
                        poolsize=(2, 2)),
            FullyConnectedLayer(n_in=20*12*12, n_out=100),
            SoftmaxLayer(n_in=100, n_out=10)],
            mini_batch_size)
        
        net.sgd(training_data, epochs, mini_batch_size, 0.1, validation_data, test_data)
    
    return net            

basic()