import numpy as np
import funkcje as fa
class Warstwa:
    def __init__(self, layer_size, activation, activation_prime, weights_init='he', input_size=None):
        self.input_size = input_size
        self.layer_size = layer_size
        self.activation = activation
        self.activation_prime = activation_prime    
        self.weights = None
        self.weights_init = weights_init
        self.bias = None
        self.input = None 
        self.velocity = None
        if self.input_size is not None:
            self.set(self.input_size)
    def initialize_weights(self):
        if self.activation ==fa.sigmoid:
             std_dev = np.sqrt(1 / self.input_size)
        elif self.activation ==fa.relu:
             std_dev = np.sqrt(2 / self.input_size)
        else:
             std_dev = 0.1
        self.weights = np.random.randn(self.input_size, self.layer_size) * std_dev
        self.velocity = np.zeros_like(self.weights)

    def set(self,input_size):
        self.input_size = input_size
        self.initialize_weights()
        self.bias = np.zeros((1, self.layer_size))
        
        
    def forward(self, X):
        self.input = X 
        self.z = np.dot(X, self.weights) + self.bias
        self.output = self.activation(self.z)
        return self.output

    def backward(self, dz, learning_rate,m):
        a_prev = self.input
        dW = np.dot(a_prev.T, dz) / m
        db = np.sum(dz, axis=0, keepdims=True) / m
        self.weights = self.update_weights(dW, learning_rate)
        self.bias -= learning_rate * db
        
    
    def update_weights(self, dW, learning_rate,momentum=0.8):
        self.weights -= learning_rate * dW  #prosta aktualizacja
        """self.velocity = momentum * self.velocity - learning_rate * dW
        return self.weights + self.velocity"""