# plik: network.py
import numpy as np
import pickle

class MLP:
    def __init__(self, learning_rate=0.01):
        self.layers = []
        self.learning_rate = learning_rate
        self.loss_func = None
        self.loss_prime = None

    def add(self, layer):
        if len(self.layers) == 0:
            if layer.input_dim is None:
                raise ValueError("Pierwsza warstwa musi mieć input_dim!")
            layer.build(layer.input_dim)
        else:
            prev_output_dim = self.layers[-1].units
            layer.build(prev_output_dim)
        self.layers.append(layer)

    def compile(self, loss_func, loss_prime):
        self.loss_func = loss_func
        self.loss_prime = loss_prime

    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def fit(self, X, y, epochs, batch_size=32, verbose=True):
        history = []
        m = X.shape[0]
        
        for epoch in range(epochs):
            # Tasowanie
            perm = np.random.permutation(m)
            X_shuffled = X[perm]
            y_shuffled = y[perm]

            # Batch training
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]
                
                # 1. Forward
                y_pred = self.forward(X_batch)
                
                # 2. Backward
                error = self.loss_prime(y_batch, y_pred)
                for layer in reversed(self.layers):
                    error = layer.backward(error, self.learning_rate)

            # Raportowanie
            if verbose:
                # Sprawdzamy błąd na próbce danych (nie całych, żeby było szybciej)
                # lub na całym zbiorze (dokładniej)
                sample_pred = self.forward(X)
                loss = self.loss_func(y, sample_pred)
                history.append(loss)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
        
        return history

    # --- POPRAWKA: Tego brakowało ---
    def predict(self, X):
        """Zwraca predykcje sieci (forward pass)."""
        return self.forward(X)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
            
    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
