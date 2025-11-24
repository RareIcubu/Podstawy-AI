import numpy as np

# --- Funkcje Aktywacji i Ich Pochodne ---

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    # Zwraca 1.0 tam, gdzie z > 0, i 0.0 w przeciwnym razie
    return (z > 0).astype(float)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    sig = sigmoid(z) # Poprawione wywołanie (bez self)
    return sig * (1 - sig)

def linear(z):
    # Aktywacja liniowa
    return z

def softmax(z):
    # Stabilna numerycznie implementacja softmax
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def derivative_one(z):
    """
    Funkcja pomocnicza zwracająca 1.
    Używana jako 'pochodna' dla warstwy wyjściowej Softmax, 
    gdy błąd jest obliczany jako (y_pred - y_true).
    """
    return 1.0

# --- Funkcja Straty ---

def compute_cross_entropy_loss(y_true, y_pred):
    """Oblicza stratę entropii krzyżowej."""
    m = y_true.shape[0]
    
    # Przycięcie wartości, aby uniknąć log(0)
    y_pred_clipped = np.clip(y_pred, 1e-9, 1 - 1e-9)
    
    loss = -np.sum(y_true * np.log(y_pred_clipped)) / m
    return loss