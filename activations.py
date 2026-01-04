import numpy as np

def tanh(z):
    return np.tanh(z)

def tanh_prime(z):
    return 1 - np.tanh(z)**2

def relu(z):
    return np.maximum(0, z)

def relu_prime(z):
    return (z > 0).astype(float)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)

def linear(z):
    """Funkcja identycznościowa (dla regresji/aproksymacji)."""
    return z

def linear_prime(z):
    """Pochodna funkcji liniowej to 1."""
    return np.ones_like(z)

def softmax(z):
    """Stabilny softmax."""
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def softmax_prime_dummy(z):
    return np.ones_like(z)
