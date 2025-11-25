import numpy as np

class Warstwa:
    def __init__(self, input_size, layer_size, activation, activation_prime, weights_init='he'):
        self.input_size = input_size
        self.layer_size = layer_size
        self.activation = activation
        self.activation_prime = activation_prime    
        self.initialize_weights(weights_init)
        self.bias = np.zeros((1, layer_size))
        self.input = None 

    def initialize_weights(self,weights_init):
        if weights_init == 'xavier':
             std_dev = np.sqrt(1 / self.input_size)
        elif weights_init == 'he':
             std_dev = np.sqrt(2 / self.input_size)
        else:
             std_dev = 0.1
        self.weights = np.random.randn(self.input_size, self.layer_size) * std_dev

    def forward(self, X):
        self.input = X 
        self.z = np.dot(X, self.weights) + self.bias
        self.output = self.activation(self.z)
        return self.output

    def backward(self, dz, learning_rate,m):
        a_prev = self.input
        dW = np.dot(a_prev.T, dz) / m
        db = np.sum(dz, axis=0, keepdims=True) / m
        self.weights -= learning_rate * dW
        self.bias -= learning_rate * db
        

            