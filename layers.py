# plik: layers.py
import numpy as np

class Dense:
    def __init__(self, units, activation, activation_prime, input_dim=None, weights_init='he'):
        self.units = units
        self.activation = activation
        self.activation_prime = activation_prime
        self.weights_init = weights_init
        self.input_dim = input_dim
        
        self.weights = None
        self.biases = None
        self.is_built = False

    def build(self, input_dim):
        self.input_dim = input_dim
        
        if self.weights_init == 'xavier':
            std_dev = np.sqrt(1 / input_dim)
        elif self.weights_init == 'he':
            std_dev = np.sqrt(2 / input_dim)
        else:
            std_dev = 0.1

        self.weights = np.random.randn(input_dim, self.units) * std_dev
        self.biases = np.zeros((1, self.units))
        self.is_built = True

    def forward(self, input_data):
        if not self.is_built:
            raise RuntimeError("Warstwa nie została zbudowana! Użyj model.add().")
            
        self.input = input_data
        self.z = np.dot(input_data, self.weights) + self.biases
        return self.activation(self.z)

    def backward(self, output_error, learning_rate):
        # Rozmiar batcha (np. 64)
        m = self.input.shape[0]
        
        # Obliczamy deltę
        delta = output_error * self.activation_prime(self.z)
        
        # --- POPRAWKA ---
        # Gradienty muszą być uśrednione po batchu (dzielimy przez m)
        # Inaczej przy dużym batchu wagi wybuchają.
        weights_grad = np.dot(self.input.T, delta) / m
        biases_grad = np.sum(delta, axis=0, keepdims=True) / m
        
        # Propagacja błędu do poprzedniej warstwy (tu nie dzielimy, to rola następnej warstwy)
        input_error = np.dot(delta, self.weights.T)

        # Aktualizacja wag
        self.weights -= learning_rate * weights_grad
        self.biases -= learning_rate * biases_grad
        
        return input_error
