#!/python

import numpy as np
import random


# sizes == number of neurons per layer
# net = Network([2, 3, 1]) two inputs, 3 hidden, one output
# biases not set for input layer
# random weights centered around 0 for initial weights with std of 1



# math helper functions
# it's not clear why these only work outside the network class and others only 
# inside it - still figuring that one out

# sigmoid function
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
    
# derivative of sigmoid function            
def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))
        

 

class Network(object):
        
    def __init__(self, sizes):    
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        
        # init with random Guassian, mean 0 stdv 1
        # self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]  #original
        
        # init with sqrt of random Guassian, mean 0, stdv 1
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    
    
    
    # cost functions
    def cost (self, a, y):
        return (a - y)
        
        
   
            
    
      
    # grab total number of correct results
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data] 
        return sum(int(x == y) for (x, y) in test_results)
        
               
    # output = sigmoid (( w * inputs) + b)    
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a
        
        
    # stochastic gradient descent, calculate and update the network as we go through data
    # eta is the learning rate
    # epoch - all of the training samples pass through in one epoch 
    # optional test data - lets you see progress but slows things down
    # mini_batch_size is the number of training examples per pass   
    def sgd ( self, training_data, epochs, mini_batch_size, eta, lmbda, test_data=None ):
            
        # if test data provided print progress, slows things down a lot
        # convert to a list, otherwise python 3 has no way to check length 
        test_data = list(test_data)   
        if test_data: n_test = len(test_data)
        
        training_data = list(training_data)
        n = len(training_data)
            
        # loop through full training set    
        for j in range(epochs):
            # shuffled training data works much better
            random.shuffle(training_data)
            
            # break the data into batches
            mini_batches = [training_data[k : k + mini_batch_size] for k in range(0, n, mini_batch_size)]
                
            # update the network once per batch 
            for mini_batch in mini_batches:
                #self.update_mini_batch(mini_batch, eta) 
                 self.update_mini_batch(mini_batch, eta, lmbda, n)
                    
            # if test data is provided, print the progress        
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))
                
     
     
     
    # update weights and biases using gradient descent on a single batch
    # eta is the learning rate
    # nabla == gradient for this batch
    def update_mini_batch(self, mini_batch, eta, lmbda, n):
     
        # temporary storage 
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
         
        # calculate the adjustment to the weights and biases 
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)   # obtain partial derivatives
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            
        # adjust the weights and biases  
        # no regularization  
        #self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        
        # add regularization
        self.weights = [(1 - eta * (lmbda/n)) * w - (eta/len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w) ]
        
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]
        
        
        
        
    # returns the gradient for the cost function
    # nabla_ contains a list of arrays, one per layer
    def backprop(self, x, y):
        
        # temporary storage
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]   # store neurode input-output
        zs = [] 
        
        
        # push inputs through the network 
        for b, w, in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b   # weights * input + b  i.e. 'weighted input'
            zs.append(z)                    # store the output
            activation = sigmoid(z)         # run output through sigmoid function    
            activations.append(activation)  # store output for next layer
            
            
        # push errors backwards through the network
        # output layer error equation - matrix form of partial derivative of cost function
        
        #delta = self.cost(activations[-1], y) * sigmoid_prime(zs[-1])      # quadratic
        delta = self.cost(activations[-1], y)                               # cross entropy


        nabla_b[-1] = delta                 # amount to change biases
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())  # amount to change weights ( error * output)
        
        
        # for each layer work backwards
        for l in range(2, self.num_layers):
            z = zs[-l]                  
            sp = sigmoid_prime(z)           # run sum of weights * input + bias through sigmoid 
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp # calculate the adjustment
            nabla_b[-l] = delta             # amount to adjust bias
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose()) # amount to adjust weights
            
        # send back the adjustments    
        return ( nabla_b, nabla_w )
        
       
    
### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net     
     