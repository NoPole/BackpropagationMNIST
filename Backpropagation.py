#!/python


# simple backpropagation network
# http://neuralnetworksanddeeplearning.com/chap1.html
# source code from "Neural Networks and Deep Learning" Nielsen
# converted from python 2 to python 3
# tweaked a few parameters


# read in the data files and format as needed
import LoadData
training_data, validation_data, test_data = LoadData.load_data_wrapper()



########## second network ######################################################
import ImprovedNetwork


# create the network
net = ImprovedNetwork.Network([784, 30, 10])  # layer sizes ( input, hidden, output )

epochs = 30        # number of passes through full data set
batch_size = 10     # size of batches, network updated once per batch
alpha = 1.0         # learning step
lmbda = 5.0         # regularization 
net.sgd(training_data, epochs, batch_size, alpha, lmbda, test_data=test_data) # train epochs, batch size, alpha



##########  first network ######################################################
"""
# create the network
import Network
net = Network.Network([784, 30, 10])  # layer sizes ( input, hidden, output )

epochs = 30        # number of passes through full data set
batch_size = 10     # size of batches, network updated once per batch
alpha = 1.0         # learning step
net.sgd(training_data, epochs, batch_size, alpha, test_data=test_data) # train epochs, batch size, alpha
"""