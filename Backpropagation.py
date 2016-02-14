#!/python


# simple backpropagation network
# http://neuralnetworksanddeeplearning.com/chap1.html
# source code from "Neural Networks and Deep Learning" Nielsen
# converted from python 2 to python 3
# tweaked a few parameters



import LoadData
training_data, validation_data, test_data = LoadData.load_data_wrapper()




import Network
net = Network.Network([784, 30, 10])
net.sgd(training_data, 100, 10, 2.0, test_data=test_data)
