#!/python

import pickle
import numpy as np

import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import downsample
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh



# activation functions
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)


# setup theano
GPU = True
if GPU:
    print("Device set to GPU")
    try: theano.config.device = 'gpu'
    except: pass    # its already set
    theano.config.floatX = 'float32'
else:
    print("Running with CPU")
    
    
    
 
# load training data
file = '../mnist.pkl'
with open(file, 'rb') as f:
    training_data, validation_data, test_data = pickle.load(f, encoding='latin-1')
    f.close()


# load data into shared variables so Theano can load it onto gpu
def shared(data):
    shared_x = theano.shared(np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
    shared_y = theano.shared(np.asarray(data[1], dtype=theano.config.floatX), borrow=True)

shared(training_data)
shared(validation_data)
shared(test_data)




# misc functions
def size(data):
    return data[0].get_value(borrow=True).shape[0]
    
    
def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams( np.random.RandomState(0).randint(999999))
    mask = srng.binomial( n=1, p=(1-p_dropout), size=layer.shape)
    return layer * T.cast(mask, theano.config.floatX)





class Network(object):

    def __init__(self, layers, mini_batch_size):

        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = T.matrix("x")
        self.y = T.ivector("y")
        
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        
        for j in range(1, len(self.layers)):
            prev_layer, layer = self.layers[j-1], self.layers[j]
            layer.set_inpt( prev_layer.output, prev_layer.output.dropout, self.mini_batch_size)
            
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

        
        
    

    def sgd(self, training_data, epochs, mini_batch_size, eta, validation_data, test_data, lmbda=0.0):

        training_x, training_y = training_data
        test_x, test_y, = test_data
        validation_x, validation_y = validation_data
        
        
        # compute number of mini batches needed
        num_training_batches = size(training_data) / mini_batch_size
        num_validation_batches = size(validation_data) / mini_batch_size
        num_test_batches = size(test_data) / mini_batch_size
        
        
        
        # cost function, regularization, gradients and updates
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self) + 0.5 * l2_norm_squared / num_training_batches
        grads = T.grad(cost, self.params)
        updates = [(param, param-eta * grad) for param, grad in zip(self.params, grads)]


        # training functions  
        i = T.lscalar()     # batch index
        train_mb = theano.function( [i], cost, updates=updates, 
                    givens={ self.x:
                             training_x[i * self.mini_batch_size: (i+1) * self.mini_batch_size], 
                             self.y:
                             training_y[i * self.mini_batch_size: (i+1) * self.mini_batch_size]
                             })
                             
        validate_mb_accuracy = theano.function ( [i], self.layers[-1].accuracy(self.y),
                                givens={ self.x:
                                          validation_x[i * self.mini_batch_size: (i+1) * self.mini_batch_size],
                                          self.y: 
                                          validation_y[i * self.mini_batch_size: (i+1) * self.mini_batch_size],
                                          })
                                          
        test_mb_accuracy = theano.function( [i], self.layers[-1].accuracy(self.y),
                            givens={ self.x:
                                      test_x[i * self.mini_batch_size: (i+1) * self.mini_batch_size],
                                      self.y:
                                      test_y[i * self.mini_batch_size: (i+1) * self.mini_batch_size]
                                      })
                                      
        self.test_mb_predictions = theano.function ( [i], self.layers[-1].y_out, 
                                    givens={ self.x:
                                             test_x[i * self.mini_batch_size: (i+1) * self.mini_batch_size]
                                    })
                                    
        
        # actual training
        best_validation_accuracy = 0.0
        
        for epoch in range(epochs):
            for mini_batch_index in range(num_training_batches):
                iteration = num_training_batches * epoch + mini_batch_index
                
                if iteration % 1000 == 0:
                    print("Training batch number {0}".format(iteration))
                    
                cost_ij = train_mb(mini_batch_index)
                
                if (iteration + 1) % num_test_batches == 0:
                    validation_accuracy = np.mean([validate_mb_accuracy(j) for j in range(num_validation_batches)])
                    print("Epoch {0}: validation accuracy {1:2%}".format(epoch, validation_accuracy))
                    
                    if validation_accuracy >= best_validation_accuracy:
                        print("Best accuracy this run.")
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        
                        if test_data:
                            test_accuracy = np.mean([test_mb_accuracy(j) for j in range(num_test_batches)])
                            print("Test accuracy {0.2%}".format(test_accuracy))
        print("Finished training")
        print("Best validation accuracy of {0:.2%} obtained at iteration {1}".format(best_validation_accuracy, best_iteration))
        print("Corresponding test accuracy of {0:2%}".format(test_accuracy))
        
        
        #########################################################################################################################
        # layer types
        
class ConvPoolLayer(object):
    
        # filter shape (number_of_filters, number_of_input_feature_maps, filter_height, filter_width)
        # image shape (mini_batch_size, number_of_input_feature_maps, image_height, image_width)
        # pool size (y, x)  
        def __init__(self, filter_shape, image_shape, poolsize=(2, 2), activation_fn=sigmoid):
                
            self.filter_shape = filter_shape
            self.image_shape = image_shape
            self.poolsize = poolsize
            self.activation_fn = activation_fn
                
            # init    
            n_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(poolsize))
            self.w = theano.shared( np.asarray(np.random.normal(loc=0, scale=np.sqrt(1.0 / n_out),
                                        size=filter_shape), dtype=theano.config.floatX), borrow=True)
            self.b = theano.shared( np.asarray(np.random.normal(loc=0, scale=np.sqrt(1.0/n_out), size=filter_shape), 
                                        dtype=theano.config.floatX), borrow=True)
            self.params = [self.w, self.b]
                
                
        def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        
            self.inpt = inpt.reshape(self.image_shape)
            conv_out = conv.conv2d( input=self.inpt, 
                                    filters=self.w, 
                                    filter_shape=self.filter_shape, 
                                    image_shape=self.image_shape )
            pooled_out = downsample.max_pool_2d( input=conv_out, ds=self.poolsize, ignore_border=True)
########### ? idk 'You cannot drop a non-broadcastable dimension' ############################################
            self.output = self.activation_fn( pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            self.output_dropout = self.output   # no dropout this layer
            
            
class FullyConnectedLayer(object):
        
        def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        
            self.n_in = n_in
            self.n_out = n_out
            self.activation_fn = activation_fn
            self.p_dropout = p_dropout
            
            # init weights and bias
            self.w = theano.shared( np.asarray(np.random.normal(loc=0.0, scale=np.sqrt(1.0/n_out), 
                                    size=(n_in, n_out)),
                                    dtype=theano.config.floatX),
                                    name='w', borrow=True)
            self.b = theano.shared( np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out, )), 
                                    dtype=theano.config.floatX), name='b', borrow=True)
            self.params = [self.w, self.b]
            
            
            
        def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
                                                
            self.inpt = inpt.reshape((mini_batch_size, self.n_in))
            self.output = self.activation_fn( (1 - self.p_dropout) * T.dot(self.inpt, self.w) + self.b)
            self.y_out = T.argmax(self.output, axis=1)
            self.inpt_dropout = dropout_layer( inpt_dropout.reshape( (mini_batch_size, self.n_in)), self.p_dropout)
            self.output_dropout = self.activation_fn( T.dot(self.inpt_dropout, self.w) + self.b)
            
        def accuracy(self, y):
            return T.mean(T.eq(y, self.y_out))
            
            
            
class SoftmaxLayer(object):
        
        
        def __init__(self, n_in, n_out, p_dropout=0.0):
            
            self.n_in = n_in
            self.n_out = n_out
            self.p_dropout = p_dropout
            
            # init weights and bias
            self.w = theano.shared(np.zeros((n_in, n_out), dtype=theano.config.floatX), 
                                    name='w', borrow=True)
            self.b = theano.shared( np.zeros((n_out,), dtype=theano.config.floatX), name='b', borrow=True)
            self.params = [self.w, self.b]
        
            
        def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
            
            self.inpt = inpt.reshape((mini_batch_size, self.n_in))
            self.output = softmax((1 - self.p_dropout) * T.dot(self.inpt, self.w) + self.b)
            self.y_out = T.argmax(self.output, axis=1)
            self.inpt_dropout = dropout_layer( inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
            self.output_dropout = softmax( T.dot(self.inpt_dropout, self.w) + self.b)
            
        def cost(self, net):
            return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])
            
        def accuracy(self, y):
            return T.mean(T.eq(y, self.y_out))
            
            
 
            
            
            
            