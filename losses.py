# plik: losses.py
import numpy as np

# --- Dla Klasyfikacji (MNIST) ---
def cross_entropy_loss(y_true, y_pred):
    """Oblicza stratę entropii krzyżowej (log loss)."""
    m = y_true.shape[0]
    # Clip zapobiega log(0)
    p = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.sum(y_true * np.log(p)) / m

def cross_entropy_prime(y_true, y_pred):
    """
    Zakładamy, że funkcja aktywacji to Softmax.
    Wtedy gradient upraszcza się do (y_pred - y_true).
    """
    return y_pred - y_true

# --- Dla Aproksymacji (Ackley) ---
def mse_loss(y_true, y_pred):
    """Błąd średniokwadratowy."""
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    """Pochodna MSE względem wyjścia sieci."""
    return 2 * (y_pred - y_true) / y_true.size
