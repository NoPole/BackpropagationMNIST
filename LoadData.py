#!/python

import pickle


import numpy as np

# load up data from file
# file was created in python 2, using encoding-'latin-1' fixes a compatibility 
# problem with opening it with python 3. idk, but it works.

def load_data():
   
    file = '../mnist.pkl'
    with open(file, 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='latin-1')
        f.close()
    
    return (training_data, validation_data, test_data)
    
    
# re-shape data
def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    
    return (training_data, validation_data, test_data)


# convert 0-9 labels to 10 zero arrays with a 1 in the correct position
def vectorized_result(j):
    
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
    
    
training_data, validation_data, test_data = load_data()
